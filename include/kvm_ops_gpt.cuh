#pragma once
// GPT operator library for the KVM megakernel framework: header-only block-cooperative device functions
//   __device__ void kvm_op_<name>(const KvmCtx& ctx, const int32_t* args)
// one per entry of kvm_ops_gpt.def, plus kvm_gpt_dispatch(). Generic and shape-parametrized: no model-specific code.
//
// Requirements on the includer
//  * KvmCtx must be declared before this header and provide:  const KvmBuf* bufs;  int32_t nbufs;  int32_t* err;
//    (err: device word, first nonzero code wins; may be null). Everything else the ops need is in this file; if the
//    executor's KvmCtx names these differently, adapt kvm_gpt_ctx_* below (the only three touch points).
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
//   attention fwd/bwd: 2e-5 (fp32 SIMT, online softmax)   decode: 2e-5   sample: exact index vs a double reference (up to
//   measure-zero CDF boundary cases)
#include <cstdint>
#include <cuda_runtime.h>
#include <mma.h>
#include <type_traits>
#include "kvm_task_abi.h"

#define KVM_GPT_THREADS 256
#define KVM_GPT_DEVI __device__ __forceinline__

// Dynamic shared memory an op needs for a launch of the op set: 80 KiB covers hd=128 attention backward; all other ops and
// attention with hd<=64 fit in 48 KiB. The executor must opt in with cudaFuncSetAttribute(MaxDynamicSharedMemorySize) if >48 KiB.
__host__ __device__ constexpr int kvm_gpt_smem_bytes(int max_hd) { return max_hd > 64 ? 80 * 1024 : 48 * 1024; }

extern __shared__ __align__(128) unsigned char kvm_gpt_smem[];

// ---- the three KvmCtx touch points ------------------------------------------------------------------------------------
KVM_GPT_DEVI const KvmBuf* kvm_gpt_ctx_bufs(const KvmCtx& c) { return c.bufs; }
KVM_GPT_DEVI int kvm_gpt_ctx_nbufs(const KvmCtx& c) { return c.nbufs; }
KVM_GPT_DEVI int32_t* kvm_gpt_ctx_err(const KvmCtx& c) { return c.err; }

enum KvmGptErr { KVM_GPT_E_BUF = 1, KVM_GPT_E_DTYPE = 2, KVM_GPT_E_RANGE = 3, KVM_GPT_E_SHAPE = 4, KVM_GPT_E_TOKEN = 5, KVM_GPT_E_KVLEN = 6 };
enum { KVM_GPT_F32 = 0, KVM_GPT_I32 = 1 };

namespace kvmgpt {

// ---- argument validation: all checks are block-uniform, so on failure every thread returns before any barrier -------------
struct Chk {
    const KvmCtx& c;
    bool bad;
    KVM_GPT_DEVI explicit Chk(const KvmCtx& cc) : c(cc), bad(false) {}
    KVM_GPT_DEVI void fail(int code) {
        if (!bad) { int32_t* e = kvm_gpt_ctx_err(c); if (e) atomicCAS(e, 0, code); }   // any thread may report (token ids fail per thread)
        bad = true;
    }
    KVM_GPT_DEVI void need(bool ok, int code = KVM_GPT_E_RANGE) { if (!ok) fail(code); }
    KVM_GPT_DEVI const KvmBuf* get(int idx, int dtype, bool opt) {
        if (idx < 0 && opt) return nullptr;
        if (idx < 0 || idx >= kvm_gpt_ctx_nbufs(c)) { fail(KVM_GPT_E_BUF); return nullptr; }
        const KvmBuf* b = kvm_gpt_ctx_bufs(c) + idx;
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
    const int V = kvm_gpt_ctx_bufs(ctx)[a[1]].rows;
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

// 64x64 output tile, BK=32, 8 warps (4 along m x 2 along n, each 16x32 = two 16x16x8 TF32 fragments). Shared tiles keep the
// global orientation (so global loads stay coalesced for all four transpose combinations); the fragment layout (row/col
// major) absorbs the transpose. Register-prefetched single stage. PREC: 3xTF32 split (a_lo*b_hi + a_hi*b_lo + a_hi*b_hi)
// restores ~fp32 accuracy.
template <bool TA, bool TB, bool PREC>
KVM_GPT_DEVI void mm_tile(const float* A, int lda, const float* B, int ldb, float* out, int ldo, const float* bias, const float* res,
                          int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, bool gelu, bool accum) {
    namespace wm = nvcuda::wmma;
    constexpr int BK = 32, TM = 64, TN = 64;
    constexpr int ASR = TA ? BK : TM, ASC = TA ? TM : BK, ALD = ASC + 4;
    constexpr int BSR = TB ? TN : BK, BSC = TB ? BK : TN, BLD = BSC + 4;
    constexpr int CLD = TN + 4, STAGE = 2304;
    using LA = std::conditional_t<TA, wm::col_major, wm::row_major>;
    using LB = std::conditional_t<TB, wm::col_major, wm::row_major>;
    float* As = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Bs = As + STAGE;
    float* Cs = As;
    const int warp = threadIdx.x >> 5, wmi = warp >> 1, wni = warp & 1;
    const bool va = aligned16(A, lda), vb = aligned16(B, ldb);
    wm::fragment<wm::accumulator, 16, 16, 8, float> acc[2];
    wm::fill_fragment(acc[0], 0.f);
    wm::fill_fragment(acc[1], 0.f);
    TileReg<ASR, ASC> ra;
    TileReg<BSR, BSC> rb;
    auto fetch = [&](int kk) {
        if (TA) ra.load(A, lda, kk, m0, K, M, va); else ra.load(A, lda, m0, kk, M, K, va);
        if (TB) rb.load(B, ldb, n0, kk, N, K, vb); else rb.load(B, ldb, kk, n0, K, N, vb);
    };
    fetch(0);
    for (int kk = 0; kk < K; kk += BK) {
        __syncthreads();
        ra.store(As, ALD);
        rb.store(Bs, BLD);
        __syncthreads();
        if (kk + BK < K) fetch(kk + BK);
#pragma unroll
        for (int ks = 0; ks < BK / 8; ks++) {
            wm::fragment<wm::matrix_a, 16, 16, 8, wm::precision::tf32, LA> ah, al;
            const float* ap = TA ? As + (ks * 8) * ALD + wmi * 16 : As + (wmi * 16) * ALD + ks * 8;
            wm::load_matrix_sync(ah, ap, ALD);
#pragma unroll
            for (int e = 0; e < ah.num_elements; e++) {
                float x = ah.x[e], h = wm::__float_to_tf32(x);
                ah.x[e] = h;
                if (PREC) al.x[e] = wm::__float_to_tf32(x - h);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wm::fragment<wm::matrix_b, 16, 16, 8, wm::precision::tf32, LB> bh, bl;
                const float* bp = TB ? Bs + (wni * 32 + j * 16) * BLD + ks * 8 : Bs + (ks * 8) * BLD + wni * 32 + j * 16;
                wm::load_matrix_sync(bh, bp, BLD);
#pragma unroll
                for (int e = 0; e < bh.num_elements; e++) {
                    float x = bh.x[e], h = wm::__float_to_tf32(x);
                    bh.x[e] = h;
                    if (PREC) bl.x[e] = wm::__float_to_tf32(x - h);
                }
                if (PREC) { wm::mma_sync(acc[j], al, bh, acc[j]); wm::mma_sync(acc[j], ah, bl, acc[j]); }
                wm::mma_sync(acc[j], ah, bh, acc[j]);
            }
        }
    }
    __syncthreads();
    wm::store_matrix_sync(Cs + (wmi * 16) * CLD + wni * 32, acc[0], CLD, wm::mem_row_major);
    wm::store_matrix_sync(Cs + (wmi * 16) * CLD + wni * 32 + 16, acc[1], CLD, wm::mem_row_major);
    __syncthreads();
    for (int e = threadIdx.x; e < TM * TN; e += KVM_GPT_THREADS) {
        int r = e / TN, c = e % TN, gm = m0 + r, gn = n0 + c;
        if (gm >= M || gn >= N) continue;
        float v = Cs[r * CLD + c] * alpha;
        if (bias) v += bias[gn];
        if (pre) pre[(size_t)gm * ldp + gn] = v;
        if (gelu) v = gelu_f(v);
        if (res) v += res[(size_t)gm * ldr + gn];
        float* o = out + (size_t)gm * ldo + gn;
        *o = accum ? *o + v : v;
    }
}

KVM_GPT_DEVI void matmul_tile(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11];
    bool ta = fl & 1, tb = fl & 2, accum = fl & 4, gelu = fl & 8, prec = fl & 16;
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~31) == 0, KVM_GPT_E_SHAPE);
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
#define KVM_GPT_MM(TA, TB, P) mm_tile<TA, TB, P>(A, lda, B, ldb, out, ldo, bias, res, ldr, pre, ldp, m0, n0, M, N, K, alpha, gelu, accum)
    if (prec) {
        if (!ta && !tb) KVM_GPT_MM(false, false, true); else if (ta && !tb) KVM_GPT_MM(true, false, true);
        else if (!ta && tb) KVM_GPT_MM(false, true, true); else KVM_GPT_MM(true, true, true);
    } else {
        if (!ta && !tb) KVM_GPT_MM(false, false, false); else if (ta && !tb) KVM_GPT_MM(true, false, false);
        else if (!ta && tb) KVM_GPT_MM(false, true, false); else KVM_GPT_MM(true, true, false);
    }
#undef KVM_GPT_MM
}

// ---- normalization: one warp per row ----------------------------------------------------------------------------------
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

// thread-per-column serial sum over rows 0..R-1 (fixed order). Columns of adjacent threads are adjacent in memory.
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
    if (k.bad) return;
    for (int c = c0 + threadIdx.x; c < c1; c += KVM_GPT_THREADS) {
        float sg = 0.f, sb = 0.f;
        for (int r = 0; r < R; r++) {
            float g = dy[(size_t)r * ldy + c], xh = (x[(size_t)r * ldx + c] - (mean ? mean[r] : 0.f)) * rstd[r];
            sg += g * xh; sb += g;
        }
        dg[c] = acc ? dg[c] + sg : sg;
        if (db) db[c] = acc ? db[c] + sb : sb;
    }
}

KVM_GPT_DEVI void colsum(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int R = a[2], c0 = a[3], c1 = a[4], acc = a[5], ld;
    k.need(R >= 0 && c0 >= 0 && c0 <= c1);
    if (k.bad) return;
    const float* x = k.mat(a[0], R, c1, &ld);
    float* o = k.fvec(a[1], c1);
    if (k.bad) return;
    for (int c = c0 + threadIdx.x; c < c1; c += KVM_GPT_THREADS) {
        float s = 0.f;
        for (int r = 0; r < R; r++) s += x[(size_t)r * ld + c];
        o[c] = acc ? o[c] + s : s;
    }
}

// ---- elementwise ------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void gelu_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[2], e1 = a[3];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) o[i] = gelu_f(x[i]);
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
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) { float v = dy[i] * gelu_grad(x[i]); dx[i] = acc ? dx[i] + v : v; }
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
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) o[i] = x[i] + y[i];
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
    int ldq, ldk, ldv, ldo, b, h, i0, T, H, Hkv;
    float scale;
};

template <int HD>
KVM_GPT_DEVI void attn_fwd_t(const AttnArgs& A) {
    using T = AT<HD>;
    float* Qs = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Ks = Qs + T::TILE; float* Vs = Ks + T::TILE; float* Ps = Vs + T::TILE;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, kvh = A.h / (A.H / A.Hkv), Tn = A.T;
    const float* qb = A.q + (size_t)A.b * Tn * A.ldq + A.h * HD;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    tile_to_smem<HD>(qb, A.ldq, A.i0, Tn, Qs);
    const int gq = A.i0 + ri;
    float m = -INFINITY, l = 0.f, acc[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) acc[c] = 0.f;
    const int kend = (min(A.i0 + 32, Tn) - 1) / 32;
    for (int kb = 0; kb <= kend; kb++) {
        __syncthreads();
        tile_to_smem<HD>(kb_, A.ldk, kb * 32, Tn, Ks);
        tile_to_smem<HD>(vb_, A.ldv, kb * 32, Tn, Vs);
        __syncthreads();
        float s[4], rm = -INFINITY;
        dot32<HD>(Qs, Ks, ri, cj, s);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            bool ok = gq < Tn && kb * 32 + cj + 8 * jj <= gq;
            s[jj] = ok ? s[jj] * A.scale : -INFINITY;
            rm = fmaxf(rm, s[jj]);
        }
        rm = fmaxf(rm, __shfl_xor_sync(0xffffffffu, rm, 1));
        rm = fmaxf(rm, __shfl_xor_sync(0xffffffffu, rm, 2));
        rm = fmaxf(rm, __shfl_xor_sync(0xffffffffu, rm, 4));
        float mn = fmaxf(m, rm), al = (mn == -INFINITY) ? 1.f : expf(m - mn), ps = 0.f;
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            float p = (s[jj] == -INFINITY) ? 0.f : expf(s[jj] - mn);
            Ps[ri * T::PS + cj + 8 * jj] = p; ps += p;
        }
        l = l * al + ps; m = mn;
#pragma unroll
        for (int c = 0; c < T::C; c++) acc[c] *= al;
        __syncthreads();
        acc32<HD, false>(acc, Ps, Vs, ri, cj);
    }
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    l += __shfl_xor_sync(0xffffffffu, l, 4);
    if (gq < Tn) {
        float inv = l > 0.f ? 1.f / l : 0.f;
        float* orow = A.o + ((size_t)A.b * Tn + gq) * A.ldo + A.h * HD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) orow[c] = acc[c] * inv;
        if (cj == 0) A.lsew[((size_t)A.b * A.H + A.h) * Tn + gq] = l > 0.f ? m + logf(l) : 0.f;
    }
}

// dQ = scale * dS K,  dS = P (dP - delta),  P = exp(scale*QK^T - lse),  dP = dO V^T. Recomputes P from lse (no stored P).
template <int HD>
KVM_GPT_DEVI void attn_dq_t(const AttnArgs& A) {
    using T = AT<HD>;
    float* Qs = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Os = Qs + T::TILE; float* Ks = Os + T::TILE; float* Vs = Ks + T::TILE; float* Ds = Vs + T::TILE;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, kvh = A.h / (A.H / A.Hkv), Tn = A.T, gq = A.i0 + ri;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    tile_to_smem<HD>(A.q + (size_t)A.b * Tn * A.ldq + A.h * HD, A.ldq, A.i0, Tn, Qs);
    tile_to_smem<HD>(A.dout + (size_t)A.b * Tn * A.ldo + A.h * HD, A.ldo, A.i0, Tn, Os);
    const size_t hrow = ((size_t)A.b * A.H + A.h) * Tn;
    const float lr = gq < Tn ? A.lse[hrow + gq] : 0.f, dr = gq < Tn ? A.delta[hrow + gq] : 0.f;
    float acc[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) acc[c] = 0.f;
    for (int kb = 0, kend = (min(A.i0 + 32, Tn) - 1) / 32; kb <= kend; kb++) {
        __syncthreads();
        tile_to_smem<HD>(kb_, A.ldk, kb * 32, Tn, Ks);
        tile_to_smem<HD>(vb_, A.ldv, kb * 32, Tn, Vs);
        __syncthreads();
        float s[4], dp[4];
        dot32<HD>(Qs, Ks, ri, cj, s);
        dot32<HD>(Os, Vs, ri, cj, dp);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            bool ok = gq < Tn && kb * 32 + cj + 8 * jj <= gq;
            Ds[ri * T::PS + cj + 8 * jj] = ok ? expf(s[jj] * A.scale - lr) * (dp[jj] - dr) : 0.f;
        }
        __syncthreads();
        acc32<HD, false>(acc, Ds, Ks, ri, cj);
    }
    if (gq < Tn) {
        float* o = A.dq + ((size_t)A.b * Tn + gq) * A.ldq + A.h * HD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) o[c] = acc[c] * A.scale;
    }
}

// dV = sum P^T dO, dK = scale * sum dS^T Q over the GQA group's heads and all queries >= the key block, fixed loop order.
template <int HD>
KVM_GPT_DEVI void attn_dkv_t(const AttnArgs& A) {
    using T = AT<HD>;
    float* Ks = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Vs = Ks + T::TILE; float* Qs = Vs + T::TILE; float* Os = Qs + T::TILE; float* Ps = Os + T::TILE; float* Ds = Ps + 32 * T::PS;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, Tn = A.T, G = A.H / A.Hkv, kvh = A.h, k0 = A.i0;
    tile_to_smem<HD>(A.k + (size_t)A.b * Tn * A.ldk + kvh * HD, A.ldk, k0, Tn, Ks);
    tile_to_smem<HD>(A.v + (size_t)A.b * Tn * A.ldv + kvh * HD, A.ldv, k0, Tn, Vs);
    float dk[T::C], dv[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) { dk[c] = 0.f; dv[c] = 0.f; }
    for (int h = kvh * G; h < (kvh + 1) * G; h++) {
        const float* qb = A.q + (size_t)A.b * Tn * A.ldq + h * HD;
        const float* ob = A.dout + (size_t)A.b * Tn * A.ldo + h * HD;
        const size_t hrow = ((size_t)A.b * A.H + h) * Tn;
        for (int q0 = k0; q0 < Tn; q0 += 32) {
            __syncthreads();
            tile_to_smem<HD>(qb, A.ldq, q0, Tn, Qs);
            tile_to_smem<HD>(ob, A.ldo, q0, Tn, Os);
            __syncthreads();
            const int gq = q0 + ri;
            const float lr = gq < Tn ? A.lse[hrow + gq] : 0.f, dr = gq < Tn ? A.delta[hrow + gq] : 0.f;
            float s[4], dp[4];
            dot32<HD>(Qs, Ks, ri, cj, s);
            dot32<HD>(Os, Vs, ri, cj, dp);
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int gk = k0 + cj + 8 * jj;
                bool ok = gq < Tn && gk <= gq;
                float p = ok ? expf(s[jj] * A.scale - lr) : 0.f;
                Ps[ri * T::PS + cj + 8 * jj] = p;
                Ds[ri * T::PS + cj + 8 * jj] = p * (dp[jj] - dr);
            }
            __syncthreads();
            acc32<HD, true>(dv, Ps, Os, ri, cj);
            acc32<HD, true>(dk, Ds, Qs, ri, cj);
        }
    }
    if (k0 + ri < Tn) {
        size_t row = (size_t)A.b * Tn + k0 + ri;
        float *odk = A.dk + row * A.ldk + kvh * HD + cj * T::C, *odv = A.dv + row * A.ldv + kvh * HD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) { odk[c] = dk[c] * A.scale; odv[c] = dv[c]; }
    }
}

#define KVM_GPT_HD_SWITCH(hd, FN, ARGS)                                                  \
    switch (hd) {                                                                        \
        case 16: FN<16>(ARGS); break;  case 32: FN<32>(ARGS); break;                     \
        case 64: FN<64>(ARGS); break;  case 128: FN<128>(ARGS); break;                   \
        default: break;                                                                  \
    }
KVM_GPT_DEVI bool hd_ok(int hd) { return hd == 16 || hd == 32 || hd == 64 || hd == 128; }

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
    KVM_GPT_HD_SWITCH(hd, attn_fwd_t, A)
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
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl); A.dq = k.mat(a[6], R, (long)A.H * hd, &A.ldq);
    if (k.bad) return;
    k.need(aligned16(A.q, A.ldq) && aligned16(A.k, A.ldk) && aligned16(A.v, A.ldv) && aligned16(A.dout, A.ldo), KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_dq_t, A)
}
KVM_GPT_DEVI void attn_bwd_dkv(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    AttnArgs A{};
    A.b = a[8]; A.h = a[9]; A.i0 = a[10]; A.T = a[11]; A.H = a[12]; A.Hkv = a[13]; int hd = a[14]; A.scale = af(a, 15);
    if (!attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, true)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl);
    A.dk = k.mat(a[6], R, (long)A.Hkv * hd, &A.ldk); A.dv = k.mat(a[7], R, (long)A.Hkv * hd, &A.ldv);
    if (k.bad) return;
    k.need(aligned16(A.q, A.ldq) && aligned16(A.k, A.ldk) && aligned16(A.v, A.ldv) && aligned16(A.dout, A.ldo), KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_dkv_t, A)
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
    const int ldc = kvm_gpt_ctx_bufs(ctx)[a[2]].ld;
    if (kvm_gpt_ctx_bufs(ctx)[a[3]].ld != ldc) k.fail(KVM_GPT_E_SHAPE);
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
    const int ldc = kvm_gpt_ctx_bufs(ctx)[a[1]].ld;
    if (kvm_gpt_ctx_bufs(ctx)[a[2]].ld != ldc) k.fail(KVM_GPT_E_SHAPE);
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
        float mx = -INFINITY;
        for (int c = lane; c < V; c += 32) mx = fmaxf(mx, row[c]);
        mx = wmax(mx);
        float s = 0.f;
        for (int c = lane; c < V; c += 32) s += expf(row[c] - mx);
        s = wsum(s);
        if (lane == 0) { float l = mx + logf(s); ls[r] = l; rl[r] = l - row[t]; }
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
        for (int c = lane; c < V; c += 32) o[c] = skip ? 0.f : (expf(row[c] - ls[r]) - (c == t ? 1.f : 0.f)) * f;
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
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) {
        float gi = g[i] * gsc, mi = b1 * m[i] + (1.f - b1) * gi, vi = b2 * v[i] + (1.f - b2) * gi * gi;
        m[i] = mi; v[i] = vi;
        p[i] = p[i] - lr * wd * p[i] - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
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

}  // namespace kvmgpt

// ---- per-op entry points and dispatcher (same X-macro convention as the executor) -------------------------------------
#define X(name, id, sig) KVM_GPT_DEVI void kvm_op_##name(const KvmCtx& ctx, const int32_t* args) { kvmgpt::name(ctx, args); }
#include "kvm_ops_gpt.def"
#undef X

// Returns false for an op id outside the GPT library (ids 32..61).
KVM_GPT_DEVI bool kvm_gpt_dispatch(const KvmCtx& ctx, int op, const int32_t* args) {
    switch (op) {
#define X(name, id, sig) case id: kvm_op_##name(ctx, args); return true;
#include "kvm_ops_gpt.def"
#undef X
        default: return false;
    }
}
