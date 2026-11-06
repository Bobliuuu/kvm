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
//   attention fwd/bwd: 2e-5 (3xTF32 tensor-core products with fp32 accumulate and online softmax; measured 1e-6..2e-5)   decode: 2e-5   sample: exact index vs a double reference (up to
//   measure-zero CDF boundary cases)
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>
#include "kvm_task_abi.h"

#define KVM_GPT_THREADS 256
#define KVM_GPT_DEVI __device__ __forceinline__
// Rows one causal attention task covers: 128 queries (attn_fwd, attn_bwd_dq: rows [q0, q0+128)) or 128 keys (attn_bwd_dkv:
// [k0, k0+128)), 16 per warp. Callers must step q0 / k0 by exactly this value: tasks that overlap write the same output rows from
// differently grouped sums (equal to rounding, not bitwise), which is a write race and breaks the reproducibility contract.
// Mirrored by ATTN_BLOCK in frontend/kvmfe/opset.py. The bidirectional attn_full_* ops keep their own 32-row blocks.
#define KVM_GPT_ATTN_ROWS 128
static_assert(KVM_GPT_THREADS / 32 * 16 == KVM_GPT_ATTN_ROWS, "attention: one m16 row tile per warp");


#ifndef KVM_GPT_BIG_BK
#define KVM_GPT_BIG_BK 16    // k per pipeline stage of the 128x128 tile (16 or 32)
#endif
#ifndef KVM_GPT_BIG_NST
#define KVM_GPT_BIG_NST 3    // pipeline stages of the 128x128 tile
#endif
// Dynamic shared memory an op needs for a launch of the op set: 104 KiB covers hd=128 attention backward/forward (6 tiles of
// 32 x 132 floats = 101 KiB); 56 KiB covers every other op (matmul 48 KiB, attention up to hd=64 at 52 KiB), so two blocks per SM still
// fit in the 164 KiB of an A100. The executor must opt in with cudaFuncSetAttribute(MaxDynamicSharedMemorySize) (it does, from
// KVM_GPT_SMEM_BYTES).
__host__ __device__ constexpr int kvm_gpt_smem_bytes(int max_hd) {
    return max_hd > 64 ? 104 * 1024 : (1536 * KVM_GPT_BIG_BK * KVM_GPT_BIG_NST > 56 * 1024 ? 1536 * KVM_GPT_BIG_BK * KVM_GPT_BIG_NST : 56 * 1024);
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
enum { KVM_GPT_F32 = 0, KVM_GPT_I32 = 1 };

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

// ---- reductions -------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI float wsum(float v) {
#pragma unroll
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
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
    float* out = k.mat(a[3], r1, d, &ldo);
    if (k.bad) return;
    const int V = ctx.bufs[a[1]].rows;
    for (int e = threadIdx.x; e < (r1 - r0) * d; e += KVM_GPT_THREADS) {
        int r = r0 + e / d, c = e % d, id = ids[r];
        float v = 0.f;
        if ((unsigned)id < (unsigned)V) v = wte[(size_t)id * ldw + c];
        else k.fail(KVM_GPT_E_TOKEN);
        if (wpe) v += wpe[(size_t)(r % T) * ldp + c];
        out[(size_t)r * ldo + c] = v;
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

// 64x64 output tile, BK=32, 3-stage cp.async pipeline (3 x 16 KiB = 48 KiB, the epilogue reuses it), 8 warps, mma.sync m16n8k8
// TF32. Warp (wm, wn, kh): 32x32 sub-tile at (wm*32, wn*32) = 2 m16 x 4 n8 fragments, reducing k8-steps {2kh, 2kh+1} of each BK=32
// slab; the two k-halves are summed in the epilogue. Shared tiles keep the global orientation (coalesced loads for all four
// transpose combinations). K-contiguous operands (A as [m][k], B as [n][k]) are fed by ldmatrix, the others by 32-bit loads.
// PREC: 3xTF32 split (a_lo*b_hi + a_hi*b_lo + a_hi*b_hi) restores ~fp32 accuracy.
template <bool TA, bool TB, bool PREC, bool BF = false>   // BF: operands rounded to bf16, m16n8k16 bf16 mma (one k16 step per slab per warp half)
KVM_GPT_DEVI void mm_tile(const float* A, int lda, const float* B, int ldb, float* out, int ldo, const float* bias, const float* res,
                          int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, bool gelu, bool accum) {
    constexpr int BK = 32, TM = 64, TN = 64, TILE = 2048, STAGE = 2 * TILE, NST = 3, CLD = TN + 8;
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
    auto issue = [&](int it) {
        float* As = sm + (it % NST) * STAGE; float* Bs = As + TILE;
        const int kk = it * BK;
        if (TA) tile_async<false, 32, 64>(As, A, lda, kk, m0, K, M, va); else tile_async<true, 64, 32>(As, A, lda, m0, kk, M, K, va);
        if (TB) tile_async<true, 64, 32>(Bs, B, ldb, n0, kk, N, K, vb); else tile_async<false, 32, 64>(Bs, B, ldb, kk, n0, K, N, vb);
    };
    issue(0); cp_commit();
    if (nk > 1) issue(1);
    cp_commit();   // always commit, empty groups included, so wait_group counts stay uniform
    for (int it = 0; it < nk; it++) {
        cp_wait<1>();
        __syncthreads();   // tile `it` landed everywhere; everyone finished the slab that used the stage refilled next
        if (it + 2 < nk) issue(it + 2);
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
        } else {
#pragma unroll
        for (int kq = 0; kq < 2; kq++) {
            const int kk = (kh * 2 + kq) * 8;
            unsigned ah[2][4], al[2][4], bh[4][2], bl[4][2];
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int mb = wmi * 32 + i * 16;
                if (!TA) ldmatrix_x4(ah[i], As + sw_kc<32>(mb + (lane & 7) + 8 * ((lane >> 3) & 1), kk + 4 * (lane >> 4)));
                else {   // A stored [k][m]: a0=(m g, k t) a1=(m g+8, k t) a2=(m g, k t+4) a3=(m g+8, k t+4)
                    ah[i][0] = __float_as_uint(As[sw_mc<64>(kk + t, mb + g)]);     ah[i][1] = __float_as_uint(As[sw_mc<64>(kk + t, mb + g + 8)]);
                    ah[i][2] = __float_as_uint(As[sw_mc<64>(kk + t + 4, mb + g)]); ah[i][3] = __float_as_uint(As[sw_mc<64>(kk + t + 4, mb + g + 8)]);
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    float x = __uint_as_float(ah[i][e]);
                    if (PREC) { unsigned h = f2tf32(x); al[i][e] = __float_as_uint(x - __uint_as_float(h)); ah[i][e] = h; }
                    else ah[i][e] = f2tf32_op(x);
                }
            }
#pragma unroll
            for (int jp = 0; jp < 2; jp++) {
                const int nb = wni * 32 + jp * 16;
                if (TB) {
                    unsigned r[4];
                    ldmatrix_x4(r, Bs + sw_kc<32>(nb + (lane & 7) + 8 * (lane >> 4), kk + 4 * ((lane >> 3) & 1)));
                    bh[2 * jp][0] = r[0]; bh[2 * jp][1] = r[1]; bh[2 * jp + 1][0] = r[2]; bh[2 * jp + 1][1] = r[3];
                } else {   // B stored [k][n]: b0=(k t, n g) b1=(k t+4, n g)
                    bh[2 * jp][0] = __float_as_uint(Bs[sw_mc<64>(kk + t, nb + g)]);          bh[2 * jp][1] = __float_as_uint(Bs[sw_mc<64>(kk + t + 4, nb + g)]);
                    bh[2 * jp + 1][0] = __float_as_uint(Bs[sw_mc<64>(kk + t, nb + 8 + g)]);  bh[2 * jp + 1][1] = __float_as_uint(Bs[sw_mc<64>(kk + t + 4, nb + 8 + g)]);
                }
#pragma unroll
                for (int jj = 0; jj < 2; jj++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                    float x = __uint_as_float(bh[2 * jp + jj][e]);
                    if (PREC) { unsigned h = f2tf32(x); bl[2 * jp + jj][e] = __float_as_uint(x - __uint_as_float(h)); bh[2 * jp + jj][e] = h; }
                    else bh[2 * jp + jj][e] = f2tf32_op(x);
                }
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (PREC) { mma_tf32(acc[i][j], al[i], bh[j][0], bh[j][1]); mma_tf32(acc[i][j], ah[i], bl[j][0], bl[j][1]); }
                    mma_tf32(acc[i][j], ah[i], bh[j][0], bh[j][1]);
                }
        }
        }
    }
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
    // Epilogue: four consecutive columns per thread, 16-byte accesses when every operand allows them.
    const bool vec = ((n0 & 3) == 0) && aligned16(out, ldo) && (!bias || aligned16(bias, 0)) && (!res || aligned16(res, ldr)) && (!pre || aligned16(pre, ldp));
    for (int q = threadIdx.x; q < TM * TN / 4; q += KVM_GPT_THREADS) {
        const int r = q / (TN / 4), c = (q % (TN / 4)) * 4, gm = m0 + r, gn = n0 + c;
        if (gm >= M || gn >= N) continue;
        const float4 p0 = *reinterpret_cast<const float4*>(sm + r * CLD + c), p1 = *reinterpret_cast<const float4*>(sm + TM * CLD + r * CLD + c);
        float v[4] = {(p0.x + p1.x) * alpha, (p0.y + p1.y) * alpha, (p0.z + p1.z) * alpha, (p0.w + p1.w) * alpha};
        if (vec && gn + 4 <= N) {
            if (bias) { const float4 b4 = *reinterpret_cast<const float4*>(bias + gn); v[0] += b4.x; v[1] += b4.y; v[2] += b4.z; v[3] += b4.w; }
            if (pre) *reinterpret_cast<float4*>(pre + (size_t)gm * ldp + gn) = make_float4(v[0], v[1], v[2], v[3]);
            if (gelu) { v[0] = gelu_f(v[0]); v[1] = gelu_f(v[1]); v[2] = gelu_f(v[2]); v[3] = gelu_f(v[3]); }
            if (res) { const float4 r4 = *reinterpret_cast<const float4*>(res + (size_t)gm * ldr + gn); v[0] += r4.x; v[1] += r4.y; v[2] += r4.z; v[3] += r4.w; }
            float4* o = reinterpret_cast<float4*>(out + (size_t)gm * ldo + gn);
            if (accum) { const float4 o4 = *o; v[0] += o4.x; v[1] += o4.y; v[2] += o4.z; v[3] += o4.w; }
            *o = make_float4(v[0], v[1], v[2], v[3]);
        } else {
            for (int i = 0; i < 4 && gn + i < N; i++) {
                float x = v[i];
                if (bias) x += bias[gn + i];
                if (pre) pre[(size_t)gm * ldp + gn + i] = x;
                if (gelu) x = gelu_f(x);
                if (res) x += res[(size_t)gm * ldr + gn + i];
                float* o = out + (size_t)gm * ldo + gn + i;
                *o = accum ? *o + x : x;
            }
        }
    }
}

// 128x128 output tile (flag 32): the same math as mm_tile with a quarter of the shared-to-L2 traffic per FLOP, which is what bounds
// the 64x64 tile. BK=16, 3-stage cp.async pipeline (3 x 16 KiB = 48 KiB), 8 warps as 2 (m) x 4 (n), each owning a 64x32 sub-tile
// = 4 m16 x 4 n8 fragments over the whole K (no split-K); the epilogue runs straight from the accumulator registers.
template <bool TA, bool TB, bool PREC, int WM, int WN, int WTN>   // WM x WN warps, each owning a 64 x WTN sub-tile; tile = 64*WM x WTN*WN
KVM_GPT_DEVI void mm_tile_big(const float* A, int lda, const float* B, int ldb, float* out, int ldo, const float* bias, const float* res,
                              int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, bool gelu, bool accum) {
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
    auto issue = [&](int it) {
        float* As = sm + (it % NST) * STAGE; float* Bs = As + ATILE;
        const int kk = it * BK;
        if (TA) tile_async<false, BK, BM>(As, A, lda, kk, m0, K, M, va); else tile_async<true, BM, BK>(As, A, lda, m0, kk, M, K, va);
        if (TB) tile_async<true, BN, BK>(Bs, B, ldb, n0, kk, N, K, vb); else tile_async<false, BK, BN>(Bs, B, ldb, kk, n0, K, N, vb);
    };
#pragma unroll
    for (int p = 0; p < NST - 1; p++) { if (p < nk) issue(p); cp_commit(); }
    for (int it = 0; it < nk; it++) {
        cp_wait<NST - 2>();
        __syncthreads();
        if (it + NST - 1 < nk) issue(it + NST - 1);
        cp_commit();
        const float* As = sm + (it % NST) * STAGE; const float* Bs = As + ATILE;
#pragma unroll
        for (int kq = 0; kq < BK / 8; kq++) {
            const int kk = kq * 8;
            unsigned ah[4][4], al[4][4], bh[NJ][2], bl[NJ][2];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int mb = wmi * 64 + i * 16;
                if (!TA) ldmatrix_x4(ah[i], As + sw_kc<BK>(mb + (lane & 7) + 8 * ((lane >> 3) & 1), kk + 4 * (lane >> 4)));
                else {
                    ah[i][0] = __float_as_uint(As[sw_mc<BM>(kk + t, mb + g)]);     ah[i][1] = __float_as_uint(As[sw_mc<BM>(kk + t, mb + g + 8)]);
                    ah[i][2] = __float_as_uint(As[sw_mc<BM>(kk + t + 4, mb + g)]); ah[i][3] = __float_as_uint(As[sw_mc<BM>(kk + t + 4, mb + g + 8)]);
                }
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    float x = __uint_as_float(ah[i][e]);
                    if (PREC) { unsigned h = f2tf32(x); al[i][e] = __float_as_uint(x - __uint_as_float(h)); ah[i][e] = h; }
                    else ah[i][e] = f2tf32_op(x);
                }
            }
#pragma unroll
            for (int jp = 0; jp < NJ / 2; jp++) {
                const int nb = wni * WTN + jp * 16;
                if (TB) {
                    unsigned r[4];
                    ldmatrix_x4(r, Bs + sw_kc<BK>(nb + (lane & 7) + 8 * (lane >> 4), kk + 4 * ((lane >> 3) & 1)));
                    bh[2 * jp][0] = r[0]; bh[2 * jp][1] = r[1]; bh[2 * jp + 1][0] = r[2]; bh[2 * jp + 1][1] = r[3];
                } else {
                    bh[2 * jp][0] = __float_as_uint(Bs[sw_mc<BN>(kk + t, nb + g)]);          bh[2 * jp][1] = __float_as_uint(Bs[sw_mc<BN>(kk + t + 4, nb + g)]);
                    bh[2 * jp + 1][0] = __float_as_uint(Bs[sw_mc<BN>(kk + t, nb + 8 + g)]);  bh[2 * jp + 1][1] = __float_as_uint(Bs[sw_mc<BN>(kk + t + 4, nb + 8 + g)]);
                }
#pragma unroll
                for (int jj = 0; jj < 2; jj++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                    float x = __uint_as_float(bh[2 * jp + jj][e]);
                    if (PREC) { unsigned h = f2tf32(x); bl[2 * jp + jj][e] = __float_as_uint(x - __uint_as_float(h)); bh[2 * jp + jj][e] = h; }
                    else bh[2 * jp + jj][e] = f2tf32_op(x);
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++) {
                    if (PREC) { mma_tf32(acc[i][j], al[i], bh[j][0], bh[j][1]); mma_tf32(acc[i][j], ah[i], bl[j][0], bl[j][1]); }
                    mma_tf32(acc[i][j], ah[i], bh[j][0], bh[j][1]);
                }
        }
    }
    cp_wait<0>();
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
            if (pre) *reinterpret_cast<float2*>(pre + (size_t)gm * ldp + gn) = make_float2(v[0], v[1]);
            if (gelu) { v[0] = gelu_f(v[0]); v[1] = gelu_f(v[1]); }
            if (res) { const float2 r2 = *reinterpret_cast<const float2*>(res + (size_t)gm * ldr + gn); v[0] += r2.x; v[1] += r2.y; }
            float2* o = reinterpret_cast<float2*>(out + (size_t)gm * ldo + gn);
            if (accum) { const float2 o2 = *o; v[0] += o2.x; v[1] += o2.y; }
            *o = make_float2(v[0], v[1]);
        } else {
            for (int e = 0; e < 2 && gn + e < N; e++) {
                float x = v[e];
                if (bias) x += bias[gn + e];
                if (pre) pre[(size_t)gm * ldp + gn + e] = x;
                if (gelu) x = gelu_f(x);
                if (res) x += res[(size_t)gm * ldr + gn + e];
                float* o = out + (size_t)gm * ldo + gn + e;
                *o = accum ? *o + x : x;
            }
        }
    }
}

KVM_GPT_DEVI void matmul_tile(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11];
    bool ta = fl & 1, tb = fl & 2, accum = fl & 4, gelu = fl & 8, prec = fl & 16, big = fl & 32, huge = fl & 64;
    k.need(!(big && huge), KVM_GPT_E_SHAPE);
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~127) == 0, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    int lda, ldb, ldo, ldr = 0, ldp = 0;
    const float* A = k.mat(a[0], ta ? K : M, ta ? M : K, &lda);
    const float* B = k.mat(a[1], tb ? N : K, tb ? K : N, &ldb);
    float* out = k.mat(a[2], M, N, &ldo);
    const float* bias = k.fvec(a[3], N, true);
    const float* res = k.mat(a[4], M, N, &ldr, true);
    float* pre = k.mat(a[5], M, N, &ldp, true);
    if (k.bad) return;
    float alpha = af(a, 12);
#define KVM_GPT_MM(F, TA, TB, P) F<TA, TB, P KVM_GPT_MM_SHAPE>(A, lda, B, ldb, out, ldo, bias, res, ldr, pre, ldp, m0, n0, M, N, K, alpha, gelu, accum)
#define KVM_GPT_MM_T(F, P)                                                                                                   \
    if (!ta && !tb) KVM_GPT_MM(F, false, false, P); else if (ta && !tb) KVM_GPT_MM(F, true, false, P);                       \
    else if (!ta && tb) KVM_GPT_MM(F, false, true, P); else KVM_GPT_MM(F, true, true, P)
#define KVM_GPT_MM_SHAPE , 2, 4, 32
    if (big) { if (prec) { KVM_GPT_MM_T(mm_tile_big, true); } else { KVM_GPT_MM_T(mm_tile_big, false); } }
#undef KVM_GPT_MM_SHAPE
#define KVM_GPT_MM_SHAPE , 4, 2, 64
    else if (huge) { if (prec) { KVM_GPT_MM_T(mm_tile_big, true); } else { KVM_GPT_MM_T(mm_tile_big, false); } }
#undef KVM_GPT_MM_SHAPE
#define KVM_GPT_MM_SHAPE
    else if (prec) { KVM_GPT_MM_T(mm_tile, true); } else { KVM_GPT_MM_T(mm_tile, false); }
#undef KVM_GPT_MM_T
#undef KVM_GPT_MM
#undef KVM_GPT_MM_SHAPE
}

// ---- normalization: one warp per row ----------------------------------------------------------------------------------
// Fast path: every lane keeps up to KVM_GPT_NORM_NV float4 of its row in registers (rows up to 128*NV = 1024 columns), so global
// memory is read once and written once. Needs d%4==0 and 16-byte aligned rows/params; otherwise the scalar path runs.
#define KVM_GPT_NORM_NV 8
KVM_GPT_DEVI bool norm_rows_vec(int d, const float* x, int ldx, const float* p0, const float* y, int ldy, const float* p1) {
    return (d & 3) == 0 && (d >> 2) <= 32 * KVM_GPT_NORM_NV && aligned16(x, ldx) && aligned16(y, ldy) && aligned16(p0, 0) && aligned16(p1, 0);
}
template <bool RMS>
KVM_GPT_DEVI void norm_fwd(const KvmCtx& ctx, const int32_t* a, int xi, int gi, int bi, int oi, int mi, int ri, int base) {
    Chk k(ctx);
    int r0 = a[base], r1 = a[base + 1], d = a[base + 2], ldx, ldo;
    float eps = af(a, base + 3);
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    const float* x = k.mat(a[xi], r1, d, &ldx);
    const float* g = k.fvec(a[gi], d);
    const float* be = RMS ? nullptr : k.fvec(a[bi], d);
    float* out = k.mat(a[oi], r1, d, &ldo);
    float* mean = RMS ? nullptr : k.fvec(a[mi], r1);
    float* rstd = k.fvec(a[ri], r1);
    if (k.bad) return;
    int lane = threadIdx.x & 31;
    if (norm_rows_vec(d, x, ldx, g, out, ldo, RMS ? g : be)) {   // one pass over global memory: the row lives in registers
        const int d4 = d >> 2;
        for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
            const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * ldx);
            float4 xv[KVM_GPT_NORM_NV];
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lane + 32 * j;
                xv[j] = c < d4 ? xr[c] : make_float4(0.f, 0.f, 0.f, 0.f);
                s += (xv[j].x + xv[j].y) + (xv[j].z + xv[j].w);
            }
            float mu = RMS ? 0.f : wsum(s) / d, v = 0.f;
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                if (lane + 32 * j < d4) {
                    float t0 = xv[j].x - mu, t1 = xv[j].y - mu, t2 = xv[j].z - mu, t3 = xv[j].w - mu;
                    v += (t0 * t0 + t1 * t1) + (t2 * t2 + t3 * t3);
                }
            }
            float rs = 1.f / sqrtf(wsum(v) / d + eps);
            float4* orow = reinterpret_cast<float4*>(out + (size_t)r * ldo);
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lane + 32 * j;
                if (c < d4) {
                    float4 gg = reinterpret_cast<const float4*>(g)[c], bb = RMS ? make_float4(0.f, 0.f, 0.f, 0.f) : reinterpret_cast<const float4*>(be)[c], o;
                    o.x = (xv[j].x - mu) * rs * gg.x + bb.x; o.y = (xv[j].y - mu) * rs * gg.y + bb.y;
                    o.z = (xv[j].z - mu) * rs * gg.z + bb.z; o.w = (xv[j].w - mu) * rs * gg.w + bb.w;
                    orow[c] = o;
                }
            }
            if (lane == 0) { rstd[r] = rs; if (!RMS) mean[r] = mu; }
        }
        return;
    }
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        const float* xr = x + (size_t)r * ldx;
        float mu = 0.f;
        if (!RMS) {
            float s = 0.f;
            for (int c = lane; c < d; c += 32) s += xr[c];
            mu = wsum(s) / d;
        }
        float v = 0.f;
        for (int c = lane; c < d; c += 32) { float t = xr[c] - mu; v += t * t; }
        float rs = 1.f / sqrtf(wsum(v) / d + eps);
        float* orow = out + (size_t)r * ldo;
        for (int c = lane; c < d; c += 32) orow[c] = (xr[c] - mu) * rs * g[c] + (RMS ? 0.f : be[c]);
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
    const float* x = k.mat(a[0], r1, d, &ldx);
    const float* dy = k.mat(a[1], r1, d, &ldy);
    const float* g = k.fvec(a[2], d);
    const float* mean = RMS ? nullptr : k.fvec(a[mi], r1);
    const float* rstd = k.fvec(a[ri], r1);
    float* dx = k.mat(a[dxi], r1, d, &ldd);
    if (k.bad) return;
    int lane = threadIdx.x & 31;
    if (norm_rows_vec(d, x, ldx, g, dx, ldd, g) && aligned16(dy, ldy)) {
        const int d4 = d >> 2;
        for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
            const float4* xr = reinterpret_cast<const float4*>(x + (size_t)r * ldx);
            const float4* dr = reinterpret_cast<const float4*>(dy + (size_t)r * ldy);
            float mu = RMS ? 0.f : mean[r], rs = rstd[r], s1 = 0.f, s2 = 0.f;
            float4 xh[KVM_GPT_NORM_NV], gc[KVM_GPT_NORM_NV];   // normalized input and dy*gamma, kept for the second pass
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lane + 32 * j;
                if (c < d4) {
                    float4 xx = xr[c], dd = dr[c], gg = reinterpret_cast<const float4*>(g)[c];
                    gc[j] = make_float4(dd.x * gg.x, dd.y * gg.y, dd.z * gg.z, dd.w * gg.w);
                    xh[j] = make_float4((xx.x - mu) * rs, (xx.y - mu) * rs, (xx.z - mu) * rs, (xx.w - mu) * rs);
                    s1 += (gc[j].x + gc[j].y) + (gc[j].z + gc[j].w);
                    s2 += (gc[j].x * xh[j].x + gc[j].y * xh[j].y) + (gc[j].z * xh[j].z + gc[j].w * xh[j].w);
                }
            }
            s1 = RMS ? 0.f : wsum(s1) / d;
            s2 = wsum(s2) / d;
            float4* o = reinterpret_cast<float4*>(dx + (size_t)r * ldd);
#pragma unroll
            for (int j = 0; j < KVM_GPT_NORM_NV; j++) {
                int c = lane + 32 * j;
                if (c < d4) {
                    float4 v;
                    v.x = rs * (gc[j].x - s1 - xh[j].x * s2); v.y = rs * (gc[j].y - s1 - xh[j].y * s2);
                    v.z = rs * (gc[j].z - s1 - xh[j].z * s2); v.w = rs * (gc[j].w - s1 - xh[j].w * s2);
                    if (acc) { float4 old = o[c]; v.x += old.x; v.y += old.y; v.z += old.z; v.w += old.w; }
                    o[c] = v;
                }
            }
        }
        return;
    }
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        const float *xr = x + (size_t)r * ldx, *dr = dy + (size_t)r * ldy;
        float mu = RMS ? 0.f : mean[r], rs = rstd[r], s1 = 0.f, s2 = 0.f;
        for (int c = lane; c < d; c += 32) {
            float gc = dr[c] * g[c], xh = (xr[c] - mu) * rs;
            s1 += gc; s2 += gc * xh;
        }
        s1 = RMS ? 0.f : wsum(s1) / d;
        s2 = wsum(s2) / d;
        float* o = dx + (size_t)r * ldd;
        for (int c = lane; c < d; c += 32) {
            float gc = dr[c] * g[c], xh = (xr[c] - mu) * rs, v = rs * (gc - s1 - xh * s2);
            o[c] = acc ? o[c] + v : v;
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
    const float* x = k.mat(a[0], R, d, &ldx);
    const float* dy = k.mat(a[1], R, d, &ldy);
    const float* mean = k.fvec(a[2], R, true);
    const float* rstd = k.fvec(a[3], R);
    float* dg = k.fvec(a[4], d);
    float* db = k.fvec(a[5], d, true);
    if (k.bad || c0 == c1) return;
    col_reduce(R, c0, c1, dg, db, acc, [&](int r, int c, float& v1, float& v2) {
        float g = dy[(size_t)r * ldy + c];
        v1 = g * ((x[(size_t)r * ldx + c] - (mean ? mean[r] : 0.f)) * rstd[r]);
        v2 = g;
    });
}

KVM_GPT_DEVI void colsum(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int R = a[2], c0 = a[3], c1 = a[4], acc = a[5], ld;
    k.need(R >= 0 && c0 >= 0 && c0 <= c1);
    if (k.bad) return;
    const float* x = k.mat(a[0], R, c1, &ld);
    float* o = k.fvec(a[1], c1);
    if (k.bad || c0 == c1) return;
    col_reduce(R, c0, c1, o, nullptr, acc, [&](int r, int c, float& v1, float& v2) { v1 = x[(size_t)r * ld + c]; v2 = 0.f; });
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
    const float* x = k.fvec(a[0], e1);
    const float* dy = k.fvec(a[1], e1);
    float* dx = k.fvec(a[2], e1);
    if (k.bad) return;
    const float* in[2] = {x, dy};
    if (acc) ew_run<2, true>(in, dx, e0, e1, [](const float4* v, int c) { return lane4(v[2], c) + lane4(v[1], c) * gelu_grad(lane4(v[0], c)); });
    else ew_run<2, false>(in, dx, e0, e1, [](const float4* v, int c) { return lane4(v[1], c) * gelu_grad(lane4(v[0], c)); });
}
KVM_GPT_DEVI void residual_add(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    const float* y = k.fvec(a[1], e1);
    float* o = k.fvec(a[2], e1);
    if (k.bad) return;
    const float* in[2] = {x, y};
    ew_run<2, false>(in, o, e0, e1, [](const float4* v, int c) { return lane4(v[0], c) + lane4(v[1], c); });
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
    const float* x = k.mat(a[0], r1, (long)H * hd, &ldx);
    float* o = k.mat(a[1], r1, (long)H * hd, &ldo);
    const int32_t* pos = k.ivec(a[2], 1, true);
    if (k.bad) return;
    int p0 = pos ? pos[0] : 0, half = hd / 2;
    for (int e = threadIdx.x; e < (r1 - r0) * H * half; e += KVM_GPT_THREADS) {
        int i = e % half, h = (e / half) % H, r = r0 + e / (half * H);
        float ang = (float)(r % T + p0) * powf(theta, -(float)(2 * i) / (float)hd), s, c;
        sincosf(ang, &s, &c);
        s *= (float)SIGN;
        const float* xr = x + (size_t)r * ldx + h * hd;
        float x1 = xr[i], x2 = xr[i + half];
        float* orow = o + (size_t)r * ldo + h * hd;
        orow[i] = x1 * c - x2 * s;
        orow[i + half] = x1 * s + x2 * c;
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
};

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
    static constexpr int LD = HD + 4, NT = HD / 8, TILE = 32 * LD, STAGE = 2 * TILE, NST = 3, NC = NT < 8 ? NT : 8;
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
KVM_GPT_DEVI void load_a_raw(float (&a)[HD / 8][4], const float* base, int ld, int row, int rmax, int t) {
    const bool v0 = row < rmax, v1 = row + 8 < rmax;
    const float* p0 = base + (size_t)row * ld + t; const float* p1 = p0 + (size_t)8 * ld;
#pragma unroll
    for (int ks = 0; ks < HD / 8; ks++) {
        a[ks][0] = v0 ? p0[ks * 8] : 0.f; a[ks][1] = v1 ? p1[ks * 8] : 0.f;
        a[ks][2] = v0 ? p0[ks * 8 + 4] : 0.f; a[ks][3] = v1 ? p1[ks * 8 + 4] : 0.f;
    }
}
// cp.async copy of a 32 x HD tile (rows row0.. of a row-major matrix, leading dimension ldg) to shared rows of stride HD+4;
// rows >= rmax are zero-filled. The padding makes every mma fragment load bank-conflict free.
template <int HD>
KVM_GPT_DEVI void tile32_async(float* dst, const float* src, int ldg, int row0, int rmax) {
    constexpr int CPR = HD / 4;
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
    constexpr int LD = M::LD, NT = M::NT, NC = M::NC;
    float* KV = reinterpret_cast<float*>(kvm_gpt_smem);   // [3 stages][K | V]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kvh = A.h / (A.H / A.Hkv), Tn = A.T, wrow = A.i0 + warp * 16, rowA = wrow + g, rowB = rowA + 8;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    const int nk = (min(A.i0 + KVM_GPT_ATTN_ROWS, Tn) + 31) / 32;   // key tiles up to and including the block's last diagonal one
    auto issue = [&](int it) {
        if (it < nk) {
            float* st = KV + (it % M::NST) * M::STAGE;
            tile32_async<HD>(st, kb_, A.ldk, it * 32, Tn);
            tile32_async<HD>(st + M::TILE, vb_, A.ldv, it * 32, Tn);
        }
        cp_commit();   // empty groups too, so the wait counts stay uniform
    };
    issue(0); issue(1);
    float aq[NT][4];
    load_a_raw<HD>(aq, A.q + (size_t)A.b * Tn * A.ldq + A.h * HD, A.ldq, rowA, Tn, t);
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
        for (int kt = 0; kt < 4; kt++) {
            unsigned ph[4], pl[4];
            acc_to_a(s[kt], ph, pl);
#pragma unroll
            for (int n0 = 0; n0 < NT; n0 += NC) {
                unsigned bh[NC][2], bl[NC][2];
                frag_kn<LD, NC>(bh, bl, Vs, kt, n0, g, t);
                mma3n<NC>(o, n0, ph, pl, bh, bl);
            }
        }
    }
    cp_wait<0>();
    if (wrow >= Tn) return;
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    float* oa = A.o + ((size_t)A.b * Tn + rowA) * A.ldo + A.h * HD + 2 * t;
    float* ob = oa + (size_t)8 * A.ldo;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        if (rowA < Tn) { oa[n * 8] = o[n][0] * inv0; oa[n * 8 + 1] = o[n][1] * inv0; }
        if (rowB < Tn) { ob[n * 8] = o[n][2] * inv1; ob[n * 8 + 1] = o[n][3] * inv1; }
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
    constexpr int LD = M::LD, NT = M::NT, NC = M::NC;
    float* KV = reinterpret_cast<float*>(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kvh = A.h / (A.H / A.Hkv), Tn = A.T, wrow = A.i0 + warp * 16, rowA = wrow + g, rowB = rowA + 8;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    const int nk = (min(A.i0 + KVM_GPT_ATTN_ROWS, Tn) + 31) / 32;
    auto issue = [&](int it) {
        if (it < nk) {
            float* st = KV + (it % M::NST) * M::STAGE;
            tile32_async<HD>(st, kb_, A.ldk, it * 32, Tn);
            tile32_async<HD>(st + M::TILE, vb_, A.ldv, it * 32, Tn);
        }
        cp_commit();
    };
    issue(0); issue(1);
    float aq[NT][4], ad[NT][4];
    load_a_raw<HD>(aq, A.q + (size_t)A.b * Tn * A.ldq + A.h * HD, A.ldq, rowA, Tn, t);
    load_a_raw<HD>(ad, A.dout + (size_t)A.b * Tn * A.ldo + A.h * HD, A.ldo, rowA, Tn, t);
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
        for (int kt = 0; kt < 4; kt++) {
            unsigned dh[4], dl[4];
            acc_to_a(s[kt], dh, dl);
#pragma unroll
            for (int n0 = 0; n0 < NT; n0 += NC) {
                unsigned bh[NC][2], bl[NC][2];
                frag_kn<LD, NC>(bh, bl, Ks, kt, n0, g, t);
                mma3n<NC>(acc, n0, dh, dl, bh, bl);
            }
        }
    }
    cp_wait<0>();
    if (wrow >= Tn) return;
    float* oa = A.dq + ((size_t)A.b * Tn + rowA) * A.lddq + A.h * HD + 2 * t;
    float* ob = oa + (size_t)8 * A.lddq;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        if (rowA < Tn) { oa[n * 8] = acc[n][0] * A.scale; oa[n * 8 + 1] = acc[n][1] * A.scale; }
        if (rowB < Tn) { ob[n * 8] = acc[n][2] * A.scale; ob[n * 8 + 1] = acc[n][3] * A.scale; }
    }
}

// dV = sum P^T dO, dK = scale * sum dS^T Q over the GQA group's heads and all queries >= the key block, fixed loop order.
// The warp's 16 keys stay resident as K and V fragments; Q and dO tiles of (head, 32 queries) stream through the pipeline. The
// products are computed transposed (S^T = K Q^T: keys are the mma rows), so P^T and dS^T come out of the accumulators already in
// the A-operand layout of dV += P^T dO and dK += dS^T Q (query order permuted inside a k8 step, as in attn_fwd_t).
template <int HD>
KVM_GPT_DEVI void attn_dkv_t(const AttnArgs& A) {
    using M = AM<HD>;
    constexpr int LD = M::LD, NT = M::NT, NC = M::NC;
    float* QO = reinterpret_cast<float*>(kvm_gpt_smem);   // [3 stages][Q | dO]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int Tn = A.T, G = A.H / A.Hkv, kvh = A.h, k0 = A.i0, wkey = k0 + warp * 16, keyA = wkey + g, keyB = keyA + 8;
    const int nq = (Tn - k0 + 31) / 32, nit = G * nq;
    auto issue = [&](int it) {
        if (it < nit) {
            const int h = kvh * G + it / nq, q0 = k0 + (it % nq) * 32;
            float* st = QO + (it % M::NST) * M::STAGE;
            tile32_async<HD>(st, A.q + (size_t)A.b * Tn * A.ldq + h * HD, A.ldq, q0, Tn);
            tile32_async<HD>(st + M::TILE, A.dout + (size_t)A.b * Tn * A.ldo + h * HD, A.ldo, q0, Tn);
        }
        cp_commit();
    };
    issue(0); issue(1);
    float ak[NT][4], av[NT][4];
    load_a_raw<HD>(ak, A.k + (size_t)A.b * Tn * A.ldk + kvh * HD, A.ldk, keyA, Tn, t);
    load_a_raw<HD>(av, A.v + (size_t)A.b * Tn * A.ldv + kvh * HD, A.ldv, keyA, Tn, t);
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
        for (int kt = 0; kt < 4; kt++) {
            unsigned ph[4], pl[4], dh[4], dl[4];
            acc_to_a(p[kt], ph, pl);
            acc_to_a(s[kt], dh, dl);
#pragma unroll
            for (int n0 = 0; n0 < NT; n0 += NC) {
                unsigned bh[NC][2], bl[NC][2];
                frag_kn<LD, NC>(bh, bl, Os, kt, n0, g, t);
                mma3n<NC>(dv, n0, ph, pl, bh, bl);
                frag_kn<LD, NC>(bh, bl, Qs, kt, n0, g, t);
                mma3n<NC>(dk, n0, dh, dl, bh, bl);
            }
        }
    }
    cp_wait<0>();
    if (wkey >= Tn) return;
    float* da = A.dk + ((size_t)A.b * Tn + keyA) * A.lddk + kvh * HD + 2 * t; float* db = da + (size_t)8 * A.lddk;
    float* va = A.dv + ((size_t)A.b * Tn + keyA) * A.lddv + kvh * HD + 2 * t; float* vb = va + (size_t)8 * A.lddv;
#pragma unroll
    for (int n = 0; n < NT; n++) {
        if (keyA < Tn) { da[n * 8] = dk[n][0] * A.scale; da[n * 8 + 1] = dk[n][1] * A.scale; va[n * 8] = dv[n][0]; va[n * 8 + 1] = dv[n][1]; }
        if (keyB < Tn) { db[n * 8] = dk[n][2] * A.scale; db[n * 8 + 1] = dk[n][3] * A.scale; vb[n * 8] = dv[n][2]; vb[n * 8 + 1] = dv[n][3]; }
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

KVM_GPT_DEVI void attn_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    AttnArgs A{};
    A.b = a[5]; A.h = a[6]; A.i0 = a[7]; A.T = a[8]; A.H = a[9]; A.Hkv = a[10]; int hd = a[11]; A.scale = af(a, 12);
    if (!attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.o = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lsew = k.fvec(a[4], (long)(A.b + 1) * A.H * A.T);
    if (k.bad) return;
    k.need(aligned16(A.q, A.ldq) && aligned16(A.k, A.ldk) && aligned16(A.v, A.ldv), KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_fwd_g, A)
}
KVM_GPT_DEVI void attn_bwd_delta(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], T = a[5], H = a[6], hd = a[7], ldo, ldd;
    k.need(r0 >= 0 && r0 <= r1 && T > 0 && H > 0 && hd > 0);
    if (k.bad) return;
    const float* o = k.mat(a[0], r1, (long)H * hd, &ldo);
    const float* d = k.mat(a[1], r1, (long)H * hd, &ldd);
    float* del = k.fvec(a[2], (long)((r1 + T - 1) / T) * H * T);
    if (k.bad) return;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int e = w; e < (r1 - r0) * H; e += KVM_GPT_THREADS / 32) {
        int r = r0 + e / H, h = e % H;
        float s = 0.f;
        for (int c = lane; c < hd; c += 32) s += o[(size_t)r * ldo + h * hd + c] * d[(size_t)r * ldd + h * hd + c];
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
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl); A.dq = k.mat(a[6], R, (long)A.H * hd, &A.lddq);
    if (k.bad) return;
    k.need(aligned16(A.q, A.ldq) && aligned16(A.k, A.ldk) && aligned16(A.v, A.ldv) && aligned16(A.dout, A.ldo), KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_dq_g, A)
}
KVM_GPT_DEVI void attn_bwd_dkv(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    AttnArgs A{};
    A.b = a[8]; A.h = a[9]; A.i0 = a[10]; A.T = a[11]; A.H = a[12]; int hd = a[13]; A.scale = af(a, 14);
    k.need(hd > 0, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    const KvmBuf* kb = k.get(a[1], KVM_GPT_F32, false);   // Hkv is not an argument (15-word task limit): Hkv = k.cols / hd
    if (k.bad) return;
    A.Hkv = kb->cols / hd;
    k.need(A.Hkv > 0 && kb->cols % hd == 0, KVM_GPT_E_SHAPE);
    if (k.bad || !attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, true)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl);
    A.dk = k.mat(a[6], R, (long)A.Hkv * hd, &A.lddk); A.dv = k.mat(a[7], R, (long)A.Hkv * hd, &A.lddv);
    if (k.bad) return;
    k.need(aligned16(A.q, A.ldq) && aligned16(A.k, A.ldk) && aligned16(A.v, A.ldv) && aligned16(A.dout, A.ldo), KVM_GPT_E_SHAPE);
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
    const float* lg = k.mat(a[0], r1, V, &ld);
    const int32_t* tg = k.ivec(a[1], r1);
    float* rl = k.fvec(a[2], r1);
    float* ls = k.fvec(a[3], r1);
    if (k.bad) return;
    int lane = threadIdx.x & 31;
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        int t = tg[r];
        if (t == ign) { if (lane == 0) { rl[r] = 0.f; ls[r] = 0.f; } continue; }
        if ((unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); continue; }
        const float* row = lg + (size_t)r * ld;
        // single pass: per-lane running (max, sum exp(x-max)), merged across the warp afterwards
        float mx = -INFINITY, sm_ = 0.f;
        auto acc1 = [&](float x) { if (x > mx) { sm_ = mx == -INFINITY ? 0.f : sm_ * fexp(mx - x); mx = x; } if (x != -INFINITY) sm_ += fexp(x - mx); };   // -inf logits contribute 0
        if (aligned16(row, ld) && (V & 3) == 0) {
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
        if (lane == 0) { float l = M + logf(s); ls[r] = l; rl[r] = l - row[t]; }
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
KVM_GPT_DEVI void softmax_ce_bwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[5], r1 = a[6], V = a[7], ign = a[8], ld, ldd;
    float sc = af(a, 9);
    k.need(r0 >= 0 && r0 <= r1 && V > 0);
    if (k.bad) return;
    const float* lg = k.mat(a[0], r1, V, &ld);
    const int32_t* tg = k.ivec(a[1], r1);
    const float* ls = k.fvec(a[2], r1);
    const float* loss = k.fvec(a[3], 2);
    float* dl = k.mat(a[4], r1, V, &ldd);
    if (k.bad) return;
    const float f = loss[1] * sc;
    for (int r = r0 + (threadIdx.x >> 5); r < r1; r += KVM_GPT_THREADS / 32) {
        int t = tg[r], lane = threadIdx.x & 31;
        const float* row = lg + (size_t)r * ld;
        float* o = dl + (size_t)r * ldd;
        bool skip = t == ign;
        if (!skip && (unsigned)t >= (unsigned)V) { k.fail(KVM_GPT_E_TOKEN); skip = true; }
        const float lr = ls[r];
        if (aligned16(row, ld) && aligned16(o, ldd) && (V & 3) == 0) {
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
