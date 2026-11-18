#pragma once
// Extra generic operator pack for the Magni VM megakernel framework: device side of mvm_ops_extra.def (ids 96..127 and 134+).
//   __device__ void mvmextra::<name>(const MvmCtx& ctx, const int32_t* args)  plus mvm_op_<name> / mvm_extra_dispatch() standalone,
//   or MVM_OP(name) wrappers inside mvm_exec.cuh. Exact semantics are the comments in mvm_ops_extra.def.
//
// Requirements on the includer: mvm_ops_gpt.cuh included BEFORE this header (this pack reuses mvmgpt::Chk, the reductions,
// matmul_tile, adamw, the attention tile helpers and the dynamic shared memory convention). Conventions are those of the gpt pack:
//  * every op runs on a block of exactly MVM_GPT_THREADS (256) threads; all argument checks are block-uniform and happen before any
//    store, failures are reported with ctx.fail(1001..1006) and the task returns without touching memory;
//  * dynamic shared memory from offset 0 (mvm_gpt_smem), never more than MVM_GPT_SMEM_BYTES; between two ops of one block the
//    executor must __syncthreads();
//  * deterministic: fixed summation orders, no float atomics, each output element has a single writer.
// Numerics: fp32 storage, fp32 math (expf/tanhf full precision); MoE aggregates (balance loss) are accumulated in double.
// matmul_tile_bf16: operands are rounded to bf16 (RNE) when read from shared memory; bf16 values are exactly representable as
// tf32, so the TF32 tensor-core mma computes exactly the bf16 x bf16 products and accumulates in fp32 (identical math to the bf16
// mma, without needing bf16 shared tiles). K is reduced exactly as in matmul_tile (two k-halves summed in the epilogue).
// MoE routing ops are single-task: expert/token loops are spread over the 256 threads but every list is built in ascending order.
// Limits: moe ops E <= 64, k <= 8 (MVM_GPT_E_SHAPE otherwise); attn_full hd in {16,32,64,128} (smem as attn_fwd).

#include "mvm_mma_bf16.cuh"

namespace mvmextra {
using mvmgpt::Chk;
using mvmgpt::af;
using mvmgpt::ld_s;
using mvmgpt::st_s;

// ---- scalar helpers ---------------------------------------------------------------------------------------------------------
MVM_GPT_DEVI float sigm(float x) { return 1.f / (1.f + expf(-x)); }
MVM_GPT_DEVI float silu_f(float x) { return x * sigm(x); }
MVM_GPT_DEVI float dsilu_f(float x) { float s = sigm(x); return s * (1.f + x * (1.f - s)); }
// Round to bfloat16 (nearest even), result in an f32 container; inf kept, NaN stays a (quiet) NaN. Integer carry handles denormals
// and overflow to inf exactly as the hardware cvt.rn.bf16.f32 does.
MVM_GPT_DEVI float bf16r(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return (u & 0x007fffffu) ? __uint_as_float((u & 0xffff0000u) | 0x00400000u) : x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
MVM_GPT_DEVI float act_f(int kind, float x) {
    switch (kind) {
        case 0: return x > 0.f ? x : 0.f;
        case 1: return silu_f(x);
        case 2: return tanhf(x);
        default: return sigm(x);
    }
}
MVM_GPT_DEVI float act_grad(int kind, float x) {
    switch (kind) {
        case 0: return x > 0.f ? 1.f : 0.f;
        case 1: return dsilu_f(x);
        case 2: { float t = tanhf(x); return 1.f - t * t; }
        default: { float s = sigm(x); return s * (1.f - s); }
    }
}

// ---- elementwise ------------------------------------------------------------------------------------------------------------
MVM_GPT_DEVI void act_fwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int kind = a[2], e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    k.need(kind >= 0 && kind <= 3, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += MVM_GPT_THREADS) o[i] = act_f(kind, x[i]);
}
MVM_GPT_DEVI void act_bwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int kind = a[3], e0 = a[4], e1 = a[5], acc = a[6];
    k.need(e0 >= 0 && e0 <= e1);
    k.need(kind >= 0 && kind <= 3, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    const float* dy = k.fvec(a[1], e1);
    float* dx = k.fvec(a[2], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += MVM_GPT_THREADS) { float v = dy[i] * act_grad(kind, x[i]); dx[i] = acc ? dx[i] + v : v; }
}
MVM_GPT_DEVI void silu_mul_fwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    bool gb, ub, ob;   // each fp32 or bf16 storage (ld_s / st_s): a decode step's SwiGLU reads the bf16 GEMV outputs with no widening copies
    const void* g = k.vec_s(a[0], e1, &gb);
    const void* u = k.vec_s(a[1], e1, &ub);
    void* o = k.vec_s(a[2], e1, &ob);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += MVM_GPT_THREADS) st_s(o, i, silu_f(ld_s(g, i, gb)) * ld_s(u, i, ub), ob);
}
// dgate = dy * up * silu'(gate); dup = dy * silu(gate)   (shared by the fused matmul_tile_silu_mul_bwd, so both agree bitwise)
MVM_GPT_DEVI void silu_mul_bwd_elem(float g, float u, float dy, float& dg, float& du) {
    dg = dy * u * dsilu_f(g);
    du = dy * silu_f(g);
}
MVM_GPT_DEVI void silu_mul_bwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[5], e1 = a[6], acc = a[7];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* g = k.fvec(a[0], e1);
    const float* u = k.fvec(a[1], e1);
    const float* dy = k.fvec(a[2], e1);
    float* dg = k.fvec(a[3], e1);
    float* du = k.fvec(a[4], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += MVM_GPT_THREADS) {
        float vg, vu;
        silu_mul_bwd_elem(g[i], u[i], dy[i], vg, vu);
        dg[i] = acc ? dg[i] + vg : vg;
        du[i] = acc ? du[i] + vu : vu;
    }
}

// ---- row softmax: one warp per row ------------------------------------------------------------------------------------------
MVM_GPT_DEVI void softmax_rows_fwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[2], r1 = a[3], d = a[4], ldx, ldo;
    float scale = af(a, 5);
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    const float* x = k.mat(a[0], r1, d, &ldx);
    float* o = k.mat(a[1], r1, d, &ldo);
    if (k.bad) return;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int r = r0 + w; r < r1; r += MVM_GPT_THREADS / 32) {
        const float* xr = x + (size_t)r * ldx;
        float* orow = o + (size_t)r * ldo;
        float m = -INFINITY;
        for (int c = lane; c < d; c += 32) m = fmaxf(m, xr[c] * scale);
        m = mvmgpt::wmax(m);
        if (m == -INFINITY) m = 0.f;
        float s = 0.f;
        for (int c = lane; c < d; c += 32) s += expf(xr[c] * scale - m);
        s = mvmgpt::wsum(s);
        for (int c = lane; c < d; c += 32) orow[c] = s > 0.f ? expf(xr[c] * scale - m) / s : 0.f;
    }
}
MVM_GPT_DEVI void softmax_rows_bwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], d = a[5], acc = a[7], ldp, ldq, ldx;
    float scale = af(a, 6);
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    const float* p = k.mat(a[0], r1, d, &ldp);
    const float* dp = k.mat(a[1], r1, d, &ldq);
    float* dx = k.mat(a[2], r1, d, &ldx);
    if (k.bad) return;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int r = r0 + w; r < r1; r += MVM_GPT_THREADS / 32) {
        const float* pr = p + (size_t)r * ldp;
        const float* dr = dp + (size_t)r * ldq;
        float* xr = dx + (size_t)r * ldx;
        float s = 0.f;
        for (int c = lane; c < d; c += 32) s += pr[c] * dr[c];
        s = mvmgpt::wsum(s);
        for (int c = lane; c < d; c += 32) { float v = scale * pr[c] * (dr[c] - s); xr[c] = acc ? xr[c] + v : v; }
    }
}

// ---- gather / scatter-add -----------------------------------------------------------------------------------------------------
MVM_GPT_DEVI void gather_rows_scaled(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[4], r1 = a[5], d = a[6], acc = a[7], i0 = a[8], lds, ldd;
    k.need(r0 >= 0 && r0 <= r1 && d > 0 && i0 >= 0);
    if (k.bad) return;
    const float* src = k.mat(a[0], 0, d, &lds);
    const int32_t* idx = k.ivec(a[1], (long)i0 + r1);
    const float* sc = k.fvec(a[2], (long)i0 + r1, true);
    float* dst = k.mat(a[3], r1, d, &ldd);
    if (k.bad) return;
    const int R = ctx.bufs[a[0]].rows;
    for (int e = threadIdx.x; e < (r1 - r0) * d; e += MVM_GPT_THREADS) {
        int r = r0 + e / d, c = e % d, j = idx[i0 + r];
        float v = 0.f;
        if ((unsigned)j < (unsigned)R) { float s = sc ? sc[i0 + r] : 1.f; v = s * src[(size_t)j * lds + c]; }
        float* o = dst + (size_t)r * ldd + c;
        *o = acc ? *o + v : v;
    }
}
MVM_GPT_DEVI void scatter_add_rows(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[4], v0 = a[5], v1 = a[6], d = a[7], acc = a[8], i0 = a[9], lds, ldd;
    k.need(N >= 0 && v0 >= 0 && v0 <= v1 && d > 0 && i0 >= 0);
    if (k.bad) return;
    const float* src = k.mat(a[0], N, d, &lds);
    const int32_t* idx = k.ivec(a[1], (long)i0 + N);
    const float* sc = k.fvec(a[2], (long)i0 + N, true);
    float* dst = k.mat(a[3], v1, d, &ldd);
    if (k.bad) return;
    // Each (dst row v, column c) pair is owned by one thread, which scans r ascending: fixed order, no atomics.
    for (int e = threadIdx.x; e < (v1 - v0) * d; e += MVM_GPT_THREADS) {
        int v = v0 + e / d, c = e % d;
        float s = 0.f;
        for (int r = 0; r < N; r++)
            if (idx[i0 + r] == v) s += (sc ? sc[i0 + r] : 1.f) * src[(size_t)r * lds + c];
        float* o = dst + (size_t)v * ldd + c;
        *o = acc ? *o + s : s;
    }
}
MVM_GPT_DEVI void row_dot(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[4], r1 = a[5], d = a[6], acc = a[7], i0 = a[8], lda, ldb, ldo;
    k.need(r0 >= 0 && r0 <= r1 && d > 0 && i0 >= 0);
    if (k.bad) return;
    const float* A = k.mat(a[0], 0, d, &lda);
    const float* B = k.mat(a[1], r1, d, &ldb);
    const int32_t* idx = k.ivec(a[2], (long)i0 + r1, true);
    float* out = k.mat(a[3], (long)i0 + r1, 1, &ldo);
    if (k.bad) return;
    const int R = ctx.bufs[a[0]].rows, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int r = r0 + w; r < r1; r += MVM_GPT_THREADS / 32) {
        int j = idx ? idx[i0 + r] : r;
        float s = 0.f;
        if ((unsigned)j < (unsigned)R) {
            const float* ar = A + (size_t)j * lda;
            const float* br = B + (size_t)r * ldb;
            for (int c = lane; c < d; c += 32) s += ar[c] * br[c];
        }
        s = mvmgpt::wsum(s);
        if (lane == 0) { float* o = out + (size_t)(i0 + r) * ldo; *o = acc ? *o + s : s; }
    }
}

// ---- mixture-of-experts routing (single task over all tokens) -----------------------------------------------------------------
MVM_GPT_DEVI bool moe_dims(Chk& k, int N, int E, int k_) {
    k.need(N >= 1);
    k.need(E >= 1 && E <= 64 && k_ >= 1 && k_ <= 8 && k_ <= E, MVM_GPT_E_SHAPE);
    return !k.bad;
}
// number of float/int elements of a buffer when used as a flat vector (rows*cols); the buffer must be dense.
MVM_GPT_DEVI long flat_n(Chk& k, int idx, int dtype) {
    const MvmBuf* b = k.get(idx, dtype, false);
    if (!b) return 0;
    if (b->rows > 1 && b->ld != b->cols) { k.fail(MVM_GPT_E_RANGE); return 0; }
    return (long)b->rows * b->cols;
}

MVM_GPT_DEVI void moe_route(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[4], E = a[5], K = a[6], C = a[7], ldp;
    if (!moe_dims(k, N, E, K)) return;
    k.need(C >= 0);
    if (k.bad) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    int32_t* sel = k.ivec(a[1], (long)N * K);
    int32_t* slot = k.ivec(a[2], (long)N * K);
    int32_t* eidx = k.ivec(a[3], (long)E * C);
    long neidx = flat_n(k, a[3], MVM_GPT_I32);
    if (k.bad) return;
    // 1. per token top-k (descending, ties -> lowest expert id): repeated selection of the first maximum among unselected experts
    for (int t = threadIdx.x; t < N; t += MVM_GPT_THREADS) {
        const float* p = probs + (size_t)t * ldp;
        uint64_t used = 0;
        for (int j = 0; j < K; j++) {
            int bi = -1;
            float bv = 0.f;
            for (int e = 0; e < E; e++) {
                if ((used >> e) & 1ull) continue;
                float v = p[e];
                if (bi < 0 || v > bv) { bi = e; bv = v; }
            }
            used |= 1ull << bi;
            sel[(size_t)t * K + j] = bi;
        }
    }
    for (long i = threadIdx.x; i < neidx; i += MVM_GPT_THREADS) eidx[i] = -1;
    __syncthreads();
    // 2. capacity: expert e owned by thread e scans (t, j) ascending; each slot[t,j] has exactly one writer
    if ((int)threadIdx.x < E) {
        const int e = threadIdx.x;
        int cnt = 0;
        for (int i = 0; i < N * K; i++) {
            if (sel[i] != e) continue;
            if (cnt < C) { slot[i] = e * C + cnt; eidx[e * C + cnt] = i / K; cnt++; }
            else slot[i] = -1;
        }
    }
}

// Z and p_j of a token; returns false if sel[t,j] is out of range (reported).
MVM_GPT_DEVI bool moe_p(const MvmCtx& ctx, const float* probs, int ldp, const int32_t* sel, int t, int K, int E, float* p) {
    for (int j = 0; j < K; j++) {
        int e = sel[(size_t)t * K + j];
        if ((unsigned)e >= (unsigned)E) { ctx.fail(MVM_GPT_E_RANGE); return false; }
        p[j] = probs[(size_t)t * ldp + e];
    }
    return true;
}

MVM_GPT_DEVI void moe_gates_fwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[4], E = a[5], K = a[6], renorm = a[7], ldp;
    if (!moe_dims(k, N, E, K)) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    const int32_t* slot = k.ivec(a[2], (long)N * K);
    long ng = flat_n(k, a[3], MVM_GPT_F32);
    if (k.bad) return;
    float* gs = (float*)ctx.bufs[a[3]].p;
    for (long i = threadIdx.x; i < ng; i += MVM_GPT_THREADS) gs[i] = 0.f;
    __syncthreads();
    for (int t = threadIdx.x; t < N; t += MVM_GPT_THREADS) {
        float p[8];
        if (!moe_p(ctx, probs, ldp, sel, t, K, E, p)) continue;
        float Z = 1.f;
        if (renorm) { Z = 0.f; for (int j = 0; j < K; j++) Z += p[j]; }
        for (int j = 0; j < K; j++) {
            int s = slot[(size_t)t * K + j];
            if (s < 0) continue;
            if (s >= ng) { ctx.fail(MVM_GPT_E_RANGE); continue; }
            gs[s] = p[j] / Z;
        }
    }
}
MVM_GPT_DEVI void moe_gates_bwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[5], E = a[6], K = a[7], renorm = a[8], acc = a[9], ldp, ldd;
    if (!moe_dims(k, N, E, K)) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    const int32_t* slot = k.ivec(a[2], (long)N * K);
    long ng = flat_n(k, a[3], MVM_GPT_F32);
    float* dprobs = k.mat(a[4], N, E, &ldd);
    if (k.bad) return;
    const float* dgs = (const float*)ctx.bufs[a[3]].p;
    if (!acc) {
        for (int i = threadIdx.x; i < N * E; i += MVM_GPT_THREADS) dprobs[(size_t)(i / E) * ldd + i % E] = 0.f;
        __syncthreads();
    }
    for (int t = threadIdx.x; t < N; t += MVM_GPT_THREADS) {
        float p[8], dg[8];
        if (!moe_p(ctx, probs, ldp, sel, t, K, E, p)) continue;
        bool ok = true;
        for (int j = 0; j < K; j++) {
            int s = slot[(size_t)t * K + j];
            if (s >= ng) { ctx.fail(MVM_GPT_E_RANGE); ok = false; }
            dg[j] = (s >= 0 && s < ng) ? dgs[s] : 0.f;
        }
        if (!ok) continue;
        float Z = 1.f, S = 0.f;
        if (renorm) { Z = 0.f; for (int j = 0; j < K; j++) { Z += p[j]; S += dg[j] * p[j]; } }
        for (int j = 0; j < K; j++) {
            float dp = renorm ? dg[j] / Z - S / (Z * Z) : dg[j];
            dprobs[(size_t)t * ldd + sel[(size_t)t * K + j]] += dp;
        }
    }
}
// per-expert selection counts (exact integers) into shared memory
MVM_GPT_DEVI void moe_counts(const MvmCtx& ctx, const int32_t* sel, int N, int E, int K, int* cnt) {
    if ((int)threadIdx.x < E) {
        int c = 0;
        for (int i = 0; i < N * K; i++) {
            int e = sel[i];
            if ((unsigned)e >= (unsigned)E) { ctx.fail(MVM_GPT_E_RANGE); break; }
            c += e == (int)threadIdx.x;
        }
        cnt[threadIdx.x] = c;
    }
}
MVM_GPT_DEVI void moe_balance_fwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[3], E = a[4], K = a[5], ldp;
    float coef = af(a, 6);
    if (!moe_dims(k, N, E, K)) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    float* loss = k.fvec(a[2], 1);
    if (k.bad) return;
    int* cnt = reinterpret_cast<int*>(mvm_gpt_smem);
    double* fp = reinterpret_cast<double*>(mvm_gpt_smem + 256);
    moe_counts(ctx, sel, N, E, K, cnt);
    if ((int)threadIdx.x < E) {
        double P = 0.0;
        for (int t = 0; t < N; t++) P += (double)probs[(size_t)t * ldp + threadIdx.x];
        P /= (double)N;
        fp[threadIdx.x] = ((double)cnt[threadIdx.x] / ((double)N * K)) * P;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int e = 0; e < E; e++) s += fp[e];
        loss[0] = (float)((double)coef * E * s);
    }
}
MVM_GPT_DEVI void moe_balance_bwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[3], E = a[4], K = a[5], acc = a[7], ldp, ldd;
    float coef = af(a, 6);
    if (!moe_dims(k, N, E, K)) return;
    k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    float* dprobs = k.mat(a[2], N, E, &ldd);
    if (k.bad) return;
    int* cnt = reinterpret_cast<int*>(mvm_gpt_smem);
    float* g = reinterpret_cast<float*>(mvm_gpt_smem + 256);
    moe_counts(ctx, sel, N, E, K, cnt);
    if ((int)threadIdx.x < E) g[threadIdx.x] = (float)((double)coef * E * ((double)cnt[threadIdx.x] / ((double)N * K)) / (double)N);
    __syncthreads();
    for (int i = threadIdx.x; i < N * E; i += MVM_GPT_THREADS) {
        float* o = dprobs + (size_t)(i / E) * ldd + i % E;
        float v = g[i % E];
        *o = acc ? *o + v : v;
    }
}

// ---- bidirectional attention (mvmgpt tile helpers, causal mask removed) --------------------------------------------------------
template <int HD>
MVM_GPT_DEVI void attn_full_fwd_t(const mvmgpt::AttnArgs& A) {
    if constexpr (HD > MVM_GPT_MAX_HD) return;
    else {
    using T = mvmgpt::AT<HD>;
    float* Qs = reinterpret_cast<float*>(mvm_gpt_smem);
    float* Ks = Qs + T::TILE; float* Vs = Ks + T::TILE; float* Ps = Vs + T::TILE;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, kvh = A.h / (A.H / A.Hkv), Tn = A.T;
    const float* qb = A.q + (size_t)A.b * Tn * A.ldq + A.h * HD;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    mvmgpt::tile_to_smem<HD>(qb, A.ldq, A.i0, Tn, Qs);
    const int gq = A.i0 + ri;
    float m = -INFINITY, l = 0.f, acc[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) acc[c] = 0.f;
    for (int kb = 0, kend = (Tn - 1) / 32; kb <= kend; kb++) {
        __syncthreads();
        mvmgpt::tile_to_smem<HD>(kb_, A.ldk, kb * 32, Tn, Ks);
        mvmgpt::tile_to_smem<HD>(vb_, A.ldv, kb * 32, Tn, Vs);
        __syncthreads();
        float s[4], rm = -INFINITY;
        mvmgpt::dot32<HD>(Qs, Ks, ri, cj, s);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            bool ok = gq < Tn && kb * 32 + cj + 8 * jj < Tn;
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
        mvmgpt::acc32<HD, false>(acc, Ps, Vs, ri, cj);
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
}
template <int HD>
MVM_GPT_DEVI void attn_full_dq_t(const mvmgpt::AttnArgs& A) {
    if constexpr (HD > MVM_GPT_MAX_HD) return;
    else {
    using T = mvmgpt::AT<HD>;
    float* Qs = reinterpret_cast<float*>(mvm_gpt_smem);
    float* Os = Qs + T::TILE; float* Ks = Os + T::TILE; float* Vs = Ks + T::TILE; float* Ds = Vs + T::TILE;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, kvh = A.h / (A.H / A.Hkv), Tn = A.T, gq = A.i0 + ri;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    mvmgpt::tile_to_smem<HD>(A.q + (size_t)A.b * Tn * A.ldq + A.h * HD, A.ldq, A.i0, Tn, Qs);
    mvmgpt::tile_to_smem<HD>(A.dout + (size_t)A.b * Tn * A.ldo + A.h * HD, A.ldo, A.i0, Tn, Os);
    const size_t hrow = ((size_t)A.b * A.H + A.h) * Tn;
    const float lr = gq < Tn ? A.lse[hrow + gq] : 0.f, dr = gq < Tn ? A.delta[hrow + gq] : 0.f;
    float acc[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) acc[c] = 0.f;
    for (int kb = 0, kend = (Tn - 1) / 32; kb <= kend; kb++) {
        __syncthreads();
        mvmgpt::tile_to_smem<HD>(kb_, A.ldk, kb * 32, Tn, Ks);
        mvmgpt::tile_to_smem<HD>(vb_, A.ldv, kb * 32, Tn, Vs);
        __syncthreads();
        float s[4], dp[4];
        mvmgpt::dot32<HD>(Qs, Ks, ri, cj, s);
        mvmgpt::dot32<HD>(Os, Vs, ri, cj, dp);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            bool ok = gq < Tn && kb * 32 + cj + 8 * jj < Tn;
            Ds[ri * T::PS + cj + 8 * jj] = ok ? expf(s[jj] * A.scale - lr) * (dp[jj] - dr) : 0.f;
        }
        __syncthreads();
        mvmgpt::acc32<HD, false>(acc, Ds, Ks, ri, cj);
    }
    if (gq < Tn) {
        float* o = A.dq + ((size_t)A.b * Tn + gq) * A.lddq + A.h * HD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) o[c] = acc[c] * A.scale;
    }
    }
}
template <int HD>
MVM_GPT_DEVI void attn_full_dkv_t(const mvmgpt::AttnArgs& A) {
    if constexpr (HD > MVM_GPT_MAX_HD) return;
    else {
    using T = mvmgpt::AT<HD>;
    float* Ks = reinterpret_cast<float*>(mvm_gpt_smem);
    float* Vs = Ks + T::TILE; float* Qs = Vs + T::TILE; float* Os = Qs + T::TILE; float* Ps = Os + T::TILE; float* Ds = Ps + 32 * T::PS;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, Tn = A.T, G = A.H / A.Hkv, kvh = A.h, k0 = A.i0;
    mvmgpt::tile_to_smem<HD>(A.k + (size_t)A.b * Tn * A.ldk + kvh * HD, A.ldk, k0, Tn, Ks);
    mvmgpt::tile_to_smem<HD>(A.v + (size_t)A.b * Tn * A.ldv + kvh * HD, A.ldv, k0, Tn, Vs);
    float dk[T::C], dv[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) { dk[c] = 0.f; dv[c] = 0.f; }
    for (int h = kvh * G; h < (kvh + 1) * G; h++) {
        const float* qb = A.q + (size_t)A.b * Tn * A.ldq + h * HD;
        const float* ob = A.dout + (size_t)A.b * Tn * A.ldo + h * HD;
        const size_t hrow = ((size_t)A.b * A.H + h) * Tn;
        for (int q0 = 0; q0 < Tn; q0 += 32) {
            __syncthreads();
            mvmgpt::tile_to_smem<HD>(qb, A.ldq, q0, Tn, Qs);
            mvmgpt::tile_to_smem<HD>(ob, A.ldo, q0, Tn, Os);
            __syncthreads();
            const int gq = q0 + ri;
            const float lr = gq < Tn ? A.lse[hrow + gq] : 0.f, dr = gq < Tn ? A.delta[hrow + gq] : 0.f;
            float s[4], dp[4];
            mvmgpt::dot32<HD>(Qs, Ks, ri, cj, s);
            mvmgpt::dot32<HD>(Os, Vs, ri, cj, dp);
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int gk = k0 + cj + 8 * jj;
                bool ok = gq < Tn && gk < Tn;
                float p = ok ? expf(s[jj] * A.scale - lr) : 0.f;
                Ps[ri * T::PS + cj + 8 * jj] = p;
                Ds[ri * T::PS + cj + 8 * jj] = p * (dp[jj] - dr);
            }
            __syncthreads();
            mvmgpt::acc32<HD, true>(dv, Ps, Os, ri, cj);
            mvmgpt::acc32<HD, true>(dk, Ds, Qs, ri, cj);
        }
    }
    if (k0 + ri < Tn) {
        size_t row = (size_t)A.b * Tn + k0 + ri;
        float *odk = A.dk + row * A.lddk + kvh * HD + cj * T::C, *odv = A.dv + row * A.lddv + kvh * HD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) { odk[c] = dk[c] * A.scale; odv[c] = dv[c]; }
    }
    }
}

MVM_GPT_DEVI void attn_full_fwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    mvmgpt::AttnArgs A{};
    A.b = a[5]; A.h = a[6]; A.i0 = a[7]; A.T = a[8]; A.H = a[9]; A.Hkv = a[10]; int hd = a[11]; A.scale = af(a, 12);
    if (!mvmgpt::attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.o = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lsew = k.fvec(a[4], (long)(A.b + 1) * A.H * A.T);
    if (k.bad) return;
    k.need(mvmgpt::aligned16(A.q, A.ldq) && mvmgpt::aligned16(A.k, A.ldk) && mvmgpt::aligned16(A.v, A.ldv), MVM_GPT_E_SHAPE);
    if (k.bad) return;
    MVM_GPT_HD_SWITCH(hd, attn_full_fwd_t, A)
}
MVM_GPT_DEVI void attn_full_bwd_dq(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    mvmgpt::AttnArgs A{};
    A.b = a[7]; A.h = a[8]; A.i0 = a[9]; A.T = a[10]; A.H = a[11]; A.Hkv = a[12]; int hd = a[13]; A.scale = af(a, 14);
    if (!mvmgpt::attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl); A.dq = k.mat(a[6], R, (long)A.H * hd, &A.lddq);
    if (k.bad) return;
    k.need(mvmgpt::aligned16(A.q, A.ldq) && mvmgpt::aligned16(A.k, A.ldk) && mvmgpt::aligned16(A.v, A.ldv) && mvmgpt::aligned16(A.dout, A.ldo),
           MVM_GPT_E_SHAPE);
    if (k.bad) return;
    MVM_GPT_HD_SWITCH(hd, attn_full_dq_t, A)
}
MVM_GPT_DEVI void attn_full_bwd_dkv(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    mvmgpt::AttnArgs A{};
    A.b = a[8]; A.h = a[9]; A.i0 = a[10]; A.T = a[11]; A.H = a[12]; int hd = a[13]; A.scale = af(a, 14);
    k.need(hd > 0, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    const MvmBuf* kb = k.get(a[1], MVM_GPT_F32, false);   // Hkv = k.cols / hd (no argument slot)
    if (k.bad) return;
    A.Hkv = kb->cols / hd;
    k.need(A.Hkv > 0 && kb->cols % hd == 0, MVM_GPT_E_SHAPE);
    if (k.bad || !mvmgpt::attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, true)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl);
    A.dk = k.mat(a[6], R, (long)A.Hkv * hd, &A.lddk); A.dv = k.mat(a[7], R, (long)A.Hkv * hd, &A.lddv);
    if (k.bad) return;
    k.need(mvmgpt::aligned16(A.q, A.ldq) && mvmgpt::aligned16(A.k, A.ldk) && mvmgpt::aligned16(A.v, A.ldv) && mvmgpt::aligned16(A.dout, A.ldo),
           MVM_GPT_E_SHAPE);
    if (k.bad) return;
    MVM_GPT_HD_SWITCH(hd, attn_full_dkv_t, A)
}

// ---- mixed precision ------------------------------------------------------------------------------------------------------------
// round_bf16 doubles as the storage conversion: x and out are each fp32 or raw bf16 (dtype 2), flat. fp32 -> fp32: rounded values in an
// fp32 container (the emulation); fp32 -> bf16: narrow; bf16 -> fp32: widen (exact); bf16 -> bf16: copy. Rounding is RNE, once per store.
MVM_GPT_DEVI void round_bf16(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[2], e1 = a[3];
    bool xb = false, ob = false;
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const void* x = k.vec_s(a[0], e1, &xb);
    void* o = k.vec_s(a[1], e1, &ob);
    if (k.bad) return;
    if (!xb && !ob) { for (int i = e0 + threadIdx.x; i < e1; i += MVM_GPT_THREADS) ((float*)o)[i] = bf16r(((const float*)x)[i]); return; }
    int i = e0;
    if ((e0 & 3) == 0 && mvmgpt::vec4_ok(x, 0, xb) && mvmgpt::vec4_ok(o, 0, ob)) {   // narrow / widen / copy 4 elements per access (the weight shadows are 38M-element casts)
        const int n4 = (e1 - e0) >> 2;
        for (int q = threadIdx.x; q < n4; q += MVM_GPT_THREADS) mvmgpt::st4(o, (size_t)e0 + 4 * q, mvmgpt::ld4(x, (size_t)e0 + 4 * q, xb), ob);
        i = e0 + n4 * 4;
    }
    for (i += threadIdx.x; i < e1; i += MVM_GPT_THREADS) mvmgpt::st_s(o, i, mvmgpt::ld_s(x, i, xb), ob);
}

// bf16 matmul: mvmgpt::mm_tile<TA, TB, false, true> (64x64 tile, 3-stage cp.async, operands rounded to bf16 when read from shared memory,
// m16n8k16 bf16 tensor-core mma with fp32 accumulate). No precise variant.
MVM_GPT_DEVI void matmul_tile_bf16(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11];
    bool ta = fl & 1, tb = fl & 2, accum = fl & 4, gelu = fl & 8;
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~31) == 0, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    // Storage dtypes: a, b, out, res, pre may each be fp32 or raw bf16 (dtype 2); bias is fp32. sd bit i set = operand i is bf16.
    int lda, ldb, ldo, ldr = 0, ldp = 0, sd = 0;
    bool fa, fb, fo, fr = false, fp = false;
    const void* A = k.mat_s(a[0], ta ? K : M, ta ? M : K, &lda, &fa);
    const void* B = k.mat_s(a[1], tb ? N : K, tb ? K : N, &ldb, &fb);
    void* out = k.mat_s(a[2], M, N, &ldo, &fo);
    const float* bias = k.fvec(a[3], N, true);
    const void* res = k.mat_s(a[4], M, N, &ldr, &fr, true);
    void* pre = k.mat_s(a[5], M, N, &ldp, &fp, true);
    if (k.bad) return;
    sd = (fa ? 1 : 0) | (fb ? 2 : 0) | (fo ? 4 : 0) | (fr ? 8 : 0) | (fp ? 16 : 0);
    if (sd & 3) {   // the bf16 loader reads uint16 elements: pointers must be 2-byte aligned (always true for cudaMalloc'd bases)
        k.need(((((uintptr_t)A | (uintptr_t)B) & 1) == 0), MVM_GPT_E_SHAPE);
        if (k.bad) return;
    }
    float alpha = af(a, 12);
#define MVM_EX_MM(TA, TB) mvmgpt::mm_tile<TA, TB, false, true>((const float*)A, lda, (const float*)B, ldb, (float*)out, ldo, bias, (const float*)res, ldr, (float*)pre, ldp, m0, n0, M, N, K, alpha, gelu, accum, sd)
    if (!ta && !tb) MVM_EX_MM(false, false); else if (ta && !tb) MVM_EX_MM(true, false);
    else if (!ta && tb) MVM_EX_MM(false, true); else MVM_EX_MM(true, true);
#undef MVM_EX_MM
}

// bf16 tensor-core GEMM, 128 x 256 output tile (mvm_mma_bf16.cuh). Same arguments and flags (1 ta, 2 tb, 4 accumulate, 8 gelu) as
// matmul_tile_bf16; the fast path needs bf16-storage a and b with 8-element aligned ld / K / base, otherwise matmul_tile_bf16 runs the
// eight 64 x 64 sub-tiles (identical semantics, slower).
MVM_GPT_DEVI void matmul_mma_bf16(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11];
    bool ta = fl & 1, tb = fl & 2, accum = fl & 4, gelu = fl & 8;
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~15) == 0, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    int lda, ldb, ldo, ldr = 0, ldp = 0;
    bool fa, fb, fo, fr = false, fp = false;
    const void* A = k.mat_s(a[0], ta ? K : M, ta ? M : K, &lda, &fa);
    const void* B = k.mat_s(a[1], tb ? N : K, tb ? K : N, &ldb, &fb);
    void* out = k.mat_s(a[2], M, N, &ldo, &fo);
    const float* bias = k.fvec(a[3], N, true);
    const void* res = k.mat_s(a[4], M, N, &ldr, &fr, true);
    void* pre = k.mat_s(a[5], M, N, &ldp, &fp, true);
    if (k.bad) return;
    const int mnA = ta ? M : K, mnB = tb ? K : N;   // contiguous extents of A and B in the global layout: whole 16-byte chunks only
    const bool fast = fa && fb && (lda & 7) == 0 && (ldb & 7) == 0 && (K & 7) == 0 && (mnA & 7) == 0 && (mnB & 7) == 0 &&
                      (((uintptr_t)A | (uintptr_t)B) & 15) == 0 && (m0 & 7) == 0 && (n0 & 7) == 0;
    if (fast) {
        const float alpha = af(a, 12);
#ifdef MVM_HOPPER_DEVICE
        {   // sm_90a: TMA + wgmma m64n256k16 (mvm_hopper.cuh); a decline falls through to the mma.sync tile
#define MVM_HOP_CALL(TA, TB) mvmhop::mm_bf16<TA, TB>((const uint16_t*)A, lda, (const uint16_t*)B, ldb, out, ldo, fo, bias, res, ldr, fr, pre, ldp, fp, m0, n0, M, N, K, alpha, gelu, accum, mvm_gpt_smem)
            const bool done = (!ta && !tb) ? MVM_HOP_CALL(false, false) : (ta && !tb) ? MVM_HOP_CALL(true, false) : (!ta && tb) ? MVM_HOP_CALL(false, true) : MVM_HOP_CALL(true, true);
#undef MVM_HOP_CALL
            if (threadIdx.x == 0) atomicAdd(&mvm_hop_count[done ? 1 : 3], 1ull);
            if (done) return;
        }
#endif
#define MVM_MMA_CALL(TA, TB) mvmgpt::mma_bf16_tile<TA, TB>((const uint16_t*)A, lda, (const uint16_t*)B, ldb, out, ldo, fo, bias, res, ldr, fr, pre, ldp, fp, m0, n0, M, N, K, alpha, gelu, accum)
        if (!ta && !tb) MVM_MMA_CALL(false, false); else if (ta && !tb) MVM_MMA_CALL(true, false);
        else if (!ta && tb) MVM_MMA_CALL(false, true); else MVM_MMA_CALL(true, true);
#undef MVM_MMA_CALL
        return;
    }
    int32_t w[13];
    for (int i = 0; i < 13; i++) w[i] = a[i];
    for (int dm = 0; dm < MVM_MMA_BM; dm += 64)
        for (int dn = 0; dn < MVM_MMA_BN; dn += 64) {
            if (m0 + dm >= M || n0 + dn >= N) continue;
            w[6] = m0 + dm; w[7] = n0 + dn;
            matmul_tile_bf16(ctx, w);
            __syncthreads();
        }
}

// ---- optimizer schedule / misc ----------------------------------------------------------------------------------------------------
MVM_GPT_DEVI void adamw_lr(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const float* lr = k.fvec(a[8], 1);
    if (k.bad) return;
    int32_t b[13];
#pragma unroll
    for (int i = 0; i < 13; i++) b[i] = a[i];
    b[8] = __float_as_int(lr[0]);
    mvmgpt::adamw(ctx, b);
}
MVM_GPT_DEVI void lr_schedule(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int kind = a[2], warmup = a[4], total = a[5];
    float base = af(a, 3), mr = af(a, 6);
    k.need(kind >= 0 && kind <= 2, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    const int32_t* step = k.ivec(a[0], 1);
    float* lr = k.fvec(a[1], 1);
    if (k.bad) return;
    if (threadIdx.x != 0) return;
    const double t = (double)step[0];
    const double warm = (warmup > 0 && step[0] <= warmup) ? t / (double)warmup : 1.0;
    double prog = (t - (double)warmup) / (double)max(total - warmup, 1);
    prog = prog < 0.0 ? 0.0 : (prog > 1.0 ? 1.0 : prog);
    double f = 1.0;
    if (kind == 1) f = (double)mr + (1.0 - (double)mr) * 0.5 * (1.0 + cos(3.14159265358979323846 * prog));
    else if (kind == 2) f = 1.0 - (1.0 - (double)mr) * prog;
    lr[0] = (float)((double)base * warm * f);
}
MVM_GPT_DEVI void scale_flat(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4];
    float alpha = af(a, 2);
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += MVM_GPT_THREADS) o[i] = alpha * x[i];
}
MVM_GPT_DEVI void add_rowtile(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], T = a[5], d = a[6], ldx, ldp, ldo;
    k.need(r0 >= 0 && r0 <= r1 && T > 0 && d > 0);
    if (k.bad) return;
    const float* x = k.mat(a[0], r1, d, &ldx);
    const float* pos = k.mat(a[1], T, d, &ldp);
    float* o = k.mat(a[2], r1, d, &ldo);
    if (k.bad) return;
    for (int e = threadIdx.x; e < (r1 - r0) * d; e += MVM_GPT_THREADS) {
        int r = r0 + e / d, c = e % d;
        o[(size_t)r * ldo + c] = x[(size_t)r * ldx + c] + pos[(size_t)(r % T) * ldp + c];
    }
}

// ---- matmul epilogue wrappers ---------------------------------------------------------------------------------------------------------
// Validate like matmul_tile (flags 4 / 8 rejected; 32 / 64 select the 128x128 / 256x128 tile), run mvmgpt::matmul_tile into `target`, barrier, then the elementwise second pass
// over the same tile. Each element is read and rewritten by the same thread.
MVM_GPT_DEVI bool mm_wrap_check(Chk& k, int abuf, int bbuf, int tbuf, int m0, int n0, int M, int N, int K, int fl, int32_t* w) {
    bool ta = fl & 1, tb = fl & 2;
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~(1 | 2 | 16 | 32 | 64)) == 0 && !((fl & 32) && (fl & 64)), MVM_GPT_E_SHAPE);   // 32 / 64: 128x128 / 256x128 tile of matmul_tile
    if (k.bad) return false;
    int ld;
    k.mat(abuf, ta ? K : M, ta ? M : K, &ld);
    k.mat(bbuf, tb ? N : K, tb ? K : N, &ld);
    k.mat(tbuf, M, N, &ld);
    if (k.bad) return false;
    w[0] = abuf; w[1] = bbuf; w[2] = tbuf; w[3] = w[4] = w[5] = -1;
    w[6] = m0; w[7] = n0; w[8] = M; w[9] = N; w[10] = K; w[11] = fl;
    return true;
}
MVM_GPT_DEVI void matmul_tile_silu_mul(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[5], n0 = a[6], M = a[7], N = a[8], K = a[9], fl = a[10], ldo, ldg, ldu;
    int32_t w[13];
    if (!mm_wrap_check(k, a[0], a[1], a[4], m0, n0, M, N, K, fl, w)) return;   // matmul result goes to `up`
    float* out = k.mat(a[2], M, N, &ldo);
    const float* gate = k.mat(a[3], M, N, &ldg);
    float* up = k.mat(a[4], M, N, &ldu);
    if (k.bad) return;
    w[12] = a[11];
    mvmgpt::matmul_tile(ctx, w);
    __syncthreads();
    const int TW = (fl & 96) ? 128 : 64, TH = (fl & 64) ? 256 : (fl & 32) ? 128 : 64;   // the wrapped matmul tile is TH x TW
    for (int e = threadIdx.x; e < TH * TW; e += MVM_GPT_THREADS) {
        int gm = m0 + e / TW, gn = n0 + e % TW;
        if (gm >= M || gn >= N) continue;
        out[(size_t)gm * ldo + gn] = silu_f(gate[(size_t)gm * ldg + gn]) * up[(size_t)gm * ldu + gn];
    }
}
MVM_GPT_DEVI void matmul_tile_silu_mul_bwd(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11], ldg, ldu, lddg, lddu;
    int32_t w[13];
    if (!mm_wrap_check(k, a[0], a[1], a[4], m0, n0, M, N, K, fl, w)) return;   // dh lands in dgate's tile
    const float* gate = k.mat(a[2], M, N, &ldg);
    const float* up = k.mat(a[3], M, N, &ldu);
    float* dgate = k.mat(a[4], M, N, &lddg);
    float* dup = k.mat(a[5], M, N, &lddu);
    if (k.bad) return;
    w[12] = a[12];
    mvmgpt::matmul_tile(ctx, w);
    __syncthreads();
    const int TW = (fl & 96) ? 128 : 64, TH = (fl & 64) ? 256 : (fl & 32) ? 128 : 64;   // the wrapped matmul tile is TH x TW
    for (int e = threadIdx.x; e < TH * TW; e += MVM_GPT_THREADS) {
        int gm = m0 + e / TW, gn = n0 + e % TW;
        if (gm >= M || gn >= N) continue;
        float* pg = dgate + (size_t)lddg * gm + gn;
        float vg, vu;
        silu_mul_bwd_elem(gate[(size_t)gm * ldg + gn], up[(size_t)gm * ldu + gn], *pg, vg, vu);
        *pg = vg;
        dup[(size_t)gm * lddu + gn] = vu;
    }
}

// ---- decode GEMV ------------------------------------------------------------------------------------------------------------
// One output row (a decode step): out[0, n] = sum_k v[k] * w[n, k] (+ res[0, n]) for n in [n0, n1), w bf16 storage [N, K] (the
// transposed serving copy: a row of w is K contiguous elements). v is the prologue of x, rounded to bfloat16 like a bf16 matmul
// operand: pro 0 v = x; pro 1 v = x * rstd * gamma (rmsnorm_fwd's row, rstd = 1 / sqrt(sum x^2 / K + eps), recomputed by every
// task from the K elements it stages anyway); pro 2 v = silu(x) * x2 (the SwiGLU). v is staged once in shared memory; each warp
// streams MVM_GEMV_R rows of w at a time with 16-byte loads, MVM_GEMV_U per row and lane in flight before any is used (Nov 16, H100:
// two per row left a task at 11 GB/s, half an SM's share of the HBM rate), and the row sums reduce in a fixed order (lane-strided,
// then the warp's xor tree): bitwise deterministic.
#ifndef MVM_GEMV_R
#define MVM_GEMV_R 4   // rows a warp streams at once
#endif
#ifndef MVM_GEMV_U
#define MVM_GEMV_U 4   // 16-byte vectors per row and lane in flight: R x U independent loads before the first use
#endif
MVM_GPT_DEVI float gemv_dot8(uint4 q, float4 a, float4 b) {
    float s = mvmgpt::bf_lo(q.x) * a.x;
    s += mvmgpt::bf_hi(q.x) * a.y;
    s += mvmgpt::bf_lo(q.y) * a.z;
    s += mvmgpt::bf_hi(q.y) * a.w;
    s += mvmgpt::bf_lo(q.z) * b.x;
    s += mvmgpt::bf_hi(q.z) * b.y;
    s += mvmgpt::bf_lo(q.w) * b.z;
    s += mvmgpt::bf_hi(q.w) * b.w;
    return s;
}
MVM_GPT_DEVI void gemv_bf16(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const int n0 = a[6], n1 = a[7], K = a[8], pro = a[9];
    const float eps = af(a, 10);
    k.need(n0 >= 0 && n0 < n1 && K > 0 && pro >= 0 && pro <= 2);
    k.need((K & 7) == 0 && (size_t)(K + 32) * 4 <= (size_t)MVM_GPT_SMEM_BYTES, MVM_GPT_E_SHAPE);   // 8-element vectors; v and the reduction in shared memory
    if (k.bad) return;
    bool xb, x2b = false, wb, ob, rb = false;
    const void* x = k.vec_s(a[0], K, &xb);
    const void* x2 = pro == 2 ? k.vec_s(a[1], K, &x2b) : nullptr;
    const float* g = pro == 1 ? k.fvec(a[2], K) : nullptr;
    int ldw, ldo, ldr;
    const void* W = k.mat_s(a[3], n1, K, &ldw, &wb);
    void* out = k.mat_s(a[4], 1, n1, &ldo, &ob);
    const void* res = k.mat_s(a[5], 1, n1, &ldr, &rb, true);
    if (k.bad) return;
    k.need(wb && (ldw & 7) == 0 && ((uintptr_t)W & 15) == 0, MVM_GPT_E_SHAPE);   // the weight stream is 16-byte vectors of bf16
    if (k.bad) return;
    float* vs = reinterpret_cast<float*>(mvm_gpt_smem);
    float* red = vs + K;
    float rs = 1.f;
    if (pro == 1) {   // rstd of the row, as rmsnorm_fwd computes it
        float ss = 0.f;
        for (int i = threadIdx.x; i < K; i += MVM_GPT_THREADS) { const float xv = ld_s(x, i, xb); ss += xv * xv; }
        ss = mvmgpt::wsum(ss);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
        __syncthreads();
        float t = 0.f;
        for (int i = 0; i < MVM_GPT_THREADS / 32; i++) t += red[i];
        rs = 1.f / sqrtf(t / K + eps);
    }
    for (int i = threadIdx.x; i < K; i += MVM_GPT_THREADS) {
        const float xv = ld_s(x, i, xb);
        const float v = pro == 1 ? xv * rs * g[i] : pro == 2 ? silu_f(xv) * ld_s(x2, i, x2b) : xv;
        vs[i] = mvmgpt::bf_hi((uint32_t)mvmgpt::bf16_bits(v) << 16);   // rounded like a bf16 operand (exact widening back)
    }
    __syncthreads();
    constexpr int R = MVM_GEMV_R, NW = MVM_GPT_THREADS / 32;
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5, nv = K >> 3;
    const uint16_t* Wb = reinterpret_cast<const uint16_t*>(W);
    for (int nb = n0 + wp * R; nb < n1; nb += NW * R) {
        float acc[R];
        const uint4* wr[R];
#pragma unroll
        for (int r = 0; r < R; r++) { acc[r] = 0.f; wr[r] = reinterpret_cast<const uint4*>(Wb + (size_t)min(nb + r, n1 - 1) * ldw); }
        constexpr int U = MVM_GEMV_U;
        int j = lane;
        for (; j + 32 * (U - 1) < nv; j += 32 * U) {   // U vectors of each of R rows per lane in flight before any is used
            uint4 q[U][R];
#pragma unroll
            for (int u = 0; u < U; u++) {
#pragma unroll
                for (int r = 0; r < R; r++) q[u][r] = __ldcs(wr[r] + j + 32 * u);
            }
#pragma unroll
            for (int u = 0; u < U; u++) {
                const int ju = j + 32 * u;
                const float4 a0 = *reinterpret_cast<const float4*>(vs + 8 * ju), b0 = *reinterpret_cast<const float4*>(vs + 8 * ju + 4);
#pragma unroll
                for (int r = 0; r < R; r++) acc[r] += gemv_dot8(q[u][r], a0, b0);
            }
        }
        for (; j < nv; j += 32) {
            const float4 a0 = *reinterpret_cast<const float4*>(vs + 8 * j), b0 = *reinterpret_cast<const float4*>(vs + 8 * j + 4);
#pragma unroll
            for (int r = 0; r < R; r++) acc[r] += gemv_dot8(__ldcs(wr[r] + j), a0, b0);
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
            const float y = mvmgpt::wsum(acc[r]);
            const int n = nb + r;
            if (lane == 0 && n < n1) st_s(out, n, res ? y + ld_s(res, n, rb) : y, ob);
        }
    }
}

#define MVM_ATTN_EMPTY_M (-3.402823466e+38f)   // m of an empty split: the most negative finite float (no inf reaches a buffer)
// ---- decode attention, split over the context ----------------------------------------------------------------------------
// attn_decode over keys [j0, j1) of sequence b's cache for the G = H / Hkv query heads of kv head kvh at once (the GQA group reads
// k and v once, not G times): split s of S, j0 = s * ceil(n / S), n = lens[b] + 1. Writes the unnormalized partial of each head h
// to part row (b * S + s) * H + h: cols [0, hd) acc = sum_j p_j v_j, col hd m = max_j score_j, col hd + 1 l = sum_j p_j with
// p_j = exp(score_j - m), score_j = scale * q_h . k_j. An empty split writes acc 0, m = MVM_ATTN_EMPTY_M, l 0.
// flags 1 (rope): q and knew are the projections BEFORE RoPE; the task rotates q (rope_fwd's angles at position lens[b]; flags 2: the
// rotated q rounded to bf16, as a bf16 rope output would be), and the split holding key n - 1 rotates knew's kv head kvh and writes it and
// vnew's row into the cache at lens[b] (kv_append, each rounded to the cache's dtype) before it reads that key. Without flag 1, knew /
// vnew are absent and q is used as is (the cache already holds key n - 1).
// Scores: one lane per key (a warp scores 32 keys at once against q in shared memory); P.V: one lane per 4 dims of a v row (coalesced),
// p broadcast by shuffle. Online softmax per warp, the 8 warps merged in a fixed order: bitwise deterministic.
MVM_GPT_DEVI void attn_decode_part(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const int b = a[7], kvh = a[8], s = a[9], S = a[10], H = a[11], Hkv = a[12], hd = a[13], Tmax = a[14], fl = a[17];
    const float scale = af(a, 15), theta = af(a, 16);
    k.need(b >= 0 && Hkv > 0 && H % Hkv == 0 && kvh >= 0 && kvh < Hkv && s >= 0 && s < S && Tmax > 0 && (fl & ~3) == 0 && (fl & 1 || !(fl & 2)));
    const int G = Hkv > 0 ? H / Hkv : 0;
    const size_t smem = (size_t)(G * hd + 2 * hd + 8 * G * 2 + 8 * G * hd) * 4;
    k.need(hd > 0 && hd <= 128 && (hd & 7) == 0 && G >= 1 && G <= 8 && smem <= (size_t)MVM_GPT_SMEM_BYTES, MVM_GPT_E_SHAPE);
    if (k.bad) return;
    const bool rot = fl & 1;
    bool qb, kb, vb, knb = false, vnb = false;
    int ldq, ldp, ldkn = 0, ldvn = 0;
    const void* q = k.mat_s(a[0], b + 1, (long)H * hd, &ldq, &qb);
    void* kc = k.mat_s(a[1], (long)(b + 1) * Hkv * Tmax, hd, nullptr, &kb);
    void* vc = k.mat_s(a[2], (long)(b + 1) * Hkv * Tmax, hd, nullptr, &vb);
    const int32_t* lens = k.ivec(a[3], b + 1);
    float* part = k.mat(a[4], (long)(b + 1) * S * H, hd + 2, &ldp);
    const void* kn = rot ? k.mat_s(a[5], b + 1, (long)Hkv * hd, &ldkn, &knb) : nullptr;
    const void* vn = rot ? k.mat_s(a[6], b + 1, (long)Hkv * hd, &ldvn, &vnb) : nullptr;
    if (k.bad) return;
    const int ldc = ctx.bufs[a[1]].ld;
    // the cache rows are read as 16-byte vectors: ld a multiple of 8 elements, both caches 16-byte aligned and of one dtype
    k.need(ctx.bufs[a[2]].ld == ldc && (ldc & 7) == 0 && kb == vb && (((uintptr_t)kc | (uintptr_t)vc) & 15) == 0, MVM_GPT_E_SHAPE);
    const int n = lens[b] + 1;
    k.need(lens[b] >= 0 && n <= Tmax, MVM_GPT_E_KVLEN);
    if (k.bad) return;
    const int chunk = (n + S - 1) / S, j0 = s * chunk, j1 = min(n, j0 + chunk), half = hd / 2;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const size_t base = (size_t)(b * Hkv + kvh) * Tmax;
    float* qs = reinterpret_cast<float*>(mvm_gpt_smem);   // [G][hd] the group's queries (rotated)
    float* sm_m = qs + G * hd + 2 * hd;                   // [8 warps][G]
    float* sm_l = sm_m + 8 * G;
    float* sm_a = sm_l + 8 * G;                           // [8 warps][G][hd]
    const float p0 = (float)lens[b];
    for (int t = threadIdx.x; t < G * half; t += MVM_GPT_THREADS) {   // stage q, rotating pairs (i, i + half) as rope_fwd does
        const int g = t / half, i = t % half;
        const size_t qo = (size_t)b * ldq + (size_t)(kvh * G + g) * hd;
        float x1 = ld_s(q, qo + i, qb), x2 = ld_s(q, qo + i + half, qb);
        if (rot) {
            float sn, cs;
            sincosf(p0 * powf(theta, -(float)(2 * i) / (float)hd), &sn, &cs);
            const float y1 = x1 * cs - x2 * sn, y2 = x1 * sn + x2 * cs;
            x1 = y1; x2 = y2;
            if (fl & 2) { x1 = mvmgpt::bf_hi((uint32_t)mvmgpt::bf16_bits(x1) << 16); x2 = mvmgpt::bf_hi((uint32_t)mvmgpt::bf16_bits(x2) << 16); }
        }
        qs[g * hd + i] = x1;
        qs[g * hd + i + half] = x2;
    }
    if (rot && j0 <= n - 1 && n - 1 < j1) {   // this split holds the new key: append it (rotated k, v) to the cache first
        const size_t dst = (base + n - 1) * ldc, ko = (size_t)b * ldkn + (size_t)kvh * hd, vo = (size_t)b * ldvn + (size_t)kvh * hd;
        for (int i = threadIdx.x; i < half; i += MVM_GPT_THREADS) {
            float sn, cs;
            sincosf(p0 * powf(theta, -(float)(2 * i) / (float)hd), &sn, &cs);
            const float x1 = ld_s(kn, ko + i, knb), x2 = ld_s(kn, ko + i + half, knb);
            st_s(kc, dst + i, x1 * cs - x2 * sn, kb);
            st_s(kc, dst + i + half, x1 * sn + x2 * cs, kb);
        }
        for (int i = threadIdx.x; i < hd; i += MVM_GPT_THREADS) st_s(vc, dst + i, ld_s(vn, vo + i, vnb), vb);
    }
    __syncthreads();   // q staged; the new cache row is visible to every thread of the block
    float m[8], l[8], acc[8][4];
#pragma unroll
    for (int g = 0; g < 8; g++) { m[g] = -INFINITY; l[g] = 0.f; acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f; }
    const int c4 = 4 * lane;
    const bool on = c4 < hd;
    for (int jb = j0 + 32 * w; jb < j1; jb += 32 * 8) {   // a warp's block of 32 keys: lane jj scores key jb + jj
        const int cnt = min(32, j1 - jb);
        float sc[8];
        {
            const int j = jb + lane;
            float d[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (lane < cnt) {
                const size_t ro = (base + j) * ldc;
                for (int c = 0; c < hd; c += 8) {
                    float kv[8];
                    if (kb) {
                        const uint4 u = *reinterpret_cast<const uint4*>((const uint16_t*)kc + ro + c);
                        kv[0] = mvmgpt::bf_lo(u.x); kv[1] = mvmgpt::bf_hi(u.x); kv[2] = mvmgpt::bf_lo(u.y); kv[3] = mvmgpt::bf_hi(u.y);
                        kv[4] = mvmgpt::bf_lo(u.z); kv[5] = mvmgpt::bf_hi(u.z); kv[6] = mvmgpt::bf_lo(u.w); kv[7] = mvmgpt::bf_hi(u.w);
                    } else {
#pragma unroll
                        for (int e = 0; e < 8; e++) kv[e] = ((const float*)kc)[ro + c + e];
                    }
#pragma unroll
                    for (int g = 0; g < 8; g++) {
                        if (g >= G) break;
                        const float* qg = qs + g * hd + c;
#pragma unroll
                        for (int e = 0; e < 8; e++) d[g] += qg[e] * kv[e];
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < 8; g++) sc[g] = lane < cnt ? d[g] * scale : -INFINITY;
        }
        float pj[8];
#pragma unroll
        for (int g = 0; g < 8; g++) {
            if (g >= G) break;
            float bm = sc[g];
#pragma unroll
            for (int o = 16; o; o >>= 1) bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, o));
            const float mn = fmaxf(m[g], bm), al = expf(m[g] - mn);   // m = -inf -> al = 0 (bm is finite: cnt >= 1)
            pj[g] = lane < cnt ? expf(sc[g] - mn) : 0.f;
            l[g] = l[g] * al + mvmgpt::wsum(pj[g]);
            acc[g][0] *= al; acc[g][1] *= al; acc[g][2] *= al; acc[g][3] *= al;
            m[g] = mn;
        }
        for (int jj = 0; jj < cnt; jj++) {   // P.V: lane owns dims c4..c4+3 of each of the block's v rows
            float vv[4] = {0.f, 0.f, 0.f, 0.f};
            if (on) {
                const float4 t = mvmgpt::ld4(vc, (base + jb + jj) * ldc + c4, vb);
                vv[0] = t.x; vv[1] = t.y; vv[2] = t.z; vv[3] = t.w;
            }
#pragma unroll
            for (int g = 0; g < 8; g++) {
                if (g >= G) break;
                const float p = __shfl_sync(0xffffffffu, pj[g], jj);
#pragma unroll
                for (int e = 0; e < 4; e++) acc[g][e] += p * vv[e];
            }
        }
    }
    for (int g = 0; g < G; g++) {
        if (lane == 0) { sm_m[w * G + g] = m[g]; sm_l[w * G + g] = l[g]; }
        if (on) {
#pragma unroll
            for (int e = 0; e < 4; e++) sm_a[((size_t)w * G + g) * hd + c4 + e] = acc[g][e];
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < G * hd; t += MVM_GPT_THREADS) {
        const int g = t / hd, cc = t % hd;
        float M = -INFINITY;
        for (int i = 0; i < 8; i++) M = fmaxf(M, sm_m[i * G + g]);
        float L = 0.f, A = 0.f;
        if (M != -INFINITY) {
            for (int i = 0; i < 8; i++) {
                const float mi = sm_m[i * G + g];
                if (mi == -INFINITY) continue;
                const float e = expf(mi - M);
                L += e * sm_l[i * G + g];
                A += e * sm_a[((size_t)i * G + g) * hd + cc];
            }
        } else {
            M = MVM_ATTN_EMPTY_M;
        }
        float* row = part + (size_t)((b * S + s) * H + kvh * G + g) * ldp;
        row[cc] = A;
        if (cc == 0) { row[hd] = M; row[hd + 1] = L; }
    }
}
// o[b, h*hd + c] = sum_s e_s acc_s[c] / sum_s e_s l_s, e_s = exp(m_s - max_s m_s) (0 for an empty split), heads [h0, h1), splits in order.
MVM_GPT_DEVI void attn_decode_merge(const MvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const int b = a[2], h0 = a[3], h1 = a[4], S = a[5], H = a[6], hd = a[7];
    k.need(b >= 0 && h0 >= 0 && h0 < h1 && h1 <= H && S > 0 && hd > 0);
    if (k.bad) return;
    bool ob;
    int ldp, ldo;
    const float* part = k.mat(a[0], (long)(b + 1) * S * H, hd + 2, &ldp);
    void* o = k.mat_s(a[1], b + 1, (long)H * hd, &ldo, &ob);
    if (k.bad) return;
    for (int t = threadIdx.x; t < (h1 - h0) * hd; t += MVM_GPT_THREADS) {
        const int h = h0 + t / hd, cc = t % hd;
        float M = MVM_ATTN_EMPTY_M;
        for (int s = 0; s < S; s++) {
            const float* row = part + (size_t)((b * S + s) * H + h) * ldp;
            if (row[hd + 1] > 0.f) M = fmaxf(M, row[hd]);
        }
        float L = 0.f, A = 0.f;
        for (int s = 0; s < S; s++) {
            const float* row = part + (size_t)((b * S + s) * H + h) * ldp;
            if (!(row[hd + 1] > 0.f)) continue;
            const float e = expf(row[hd] - M);
            L += e * row[hd + 1];
            A += e * row[cc];
        }
        st_s(o, (size_t)b * ldo + (size_t)h * hd + cc, A / L, ob);
    }
}

}  // namespace mvmextra

// ---- per-op entry points (same X-macro convention as the gpt pack) -------------------------------------------------------------------
#ifdef MVM_OP
#define X(name, id, sig) MVM_OP(name) { mvmextra::name(c, a); }
#include "mvm_ops_extra.def"
#undef X
#else
#define X(name, id, sig) MVM_GPT_DEVI void mvm_op_##name(const MvmCtx& ctx, const int32_t* args) { mvmextra::name(ctx, args); }
#include "mvm_ops_extra.def"
#undef X
// Returns false for an op id outside the extra pack.
MVM_GPT_DEVI bool mvm_extra_dispatch(const MvmCtx& ctx, int op, const int32_t* args) {
    switch (op) {
#define X(name, id, sig) case id: mvm_op_##name(ctx, args); return true;
#include "mvm_ops_extra.def"
#undef X
        default: return false;
    }
}
#endif
