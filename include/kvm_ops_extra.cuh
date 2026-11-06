#pragma once
// Extra generic operator pack for the KVM megakernel framework: device side of kvm_ops_extra.def (ids 96..121).
//   __device__ void kvmextra::<name>(const KvmCtx& ctx, const int32_t* args)  plus kvm_op_<name> / kvm_extra_dispatch() standalone,
//   or KVM_OP(name) wrappers inside kvm_exec.cuh. Exact semantics are the comments in kvm_ops_extra.def.
//
// Requirements on the includer: kvm_ops_gpt.cuh included BEFORE this header (this pack reuses kvmgpt::Chk, the reductions,
// matmul_tile, adamw, the attention tile helpers and the dynamic shared memory convention). Conventions are those of the gpt pack:
//  * every op runs on a block of exactly KVM_GPT_THREADS (256) threads; all argument checks are block-uniform and happen before any
//    store, failures are reported with ctx.fail(1001..1006) and the task returns without touching memory;
//  * dynamic shared memory from offset 0 (kvm_gpt_smem), never more than KVM_GPT_SMEM_BYTES; between two ops of one block the
//    executor must __syncthreads();
//  * deterministic: fixed summation orders, no float atomics, each output element has a single writer.
// Numerics: fp32 storage, fp32 math (expf/tanhf full precision); MoE aggregates (balance loss) are accumulated in double.
// matmul_tile_bf16: operands are rounded to bf16 (RNE) when read from shared memory; bf16 values are exactly representable as
// tf32, so the TF32 tensor-core mma computes exactly the bf16 x bf16 products and accumulates in fp32 (identical math to the bf16
// mma, without needing bf16 shared tiles). K is reduced exactly as in matmul_tile (two k-halves summed in the epilogue).
// MoE routing ops are single-task: expert/token loops are spread over the 256 threads but every list is built in ascending order.
// Limits: moe ops E <= 64, k <= 8 (KVM_GPT_E_SHAPE otherwise); attn_full hd in {16,32,64,128} (smem as attn_fwd).

namespace kvmextra {
using kvmgpt::Chk;
using kvmgpt::af;

// ---- scalar helpers ---------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI float sigm(float x) { return 1.f / (1.f + expf(-x)); }
KVM_GPT_DEVI float silu_f(float x) { return x * sigm(x); }
KVM_GPT_DEVI float dsilu_f(float x) { float s = sigm(x); return s * (1.f + x * (1.f - s)); }
// Round to bfloat16 (nearest even), result in an f32 container; inf kept, NaN stays a (quiet) NaN. Integer carry handles denormals
// and overflow to inf exactly as the hardware cvt.rn.bf16.f32 does.
KVM_GPT_DEVI float bf16r(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7f800000u) == 0x7f800000u) return (u & 0x007fffffu) ? __uint_as_float((u & 0xffff0000u) | 0x00400000u) : x;
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
KVM_GPT_DEVI float act_f(int kind, float x) {
    switch (kind) {
        case 0: return x > 0.f ? x : 0.f;
        case 1: return silu_f(x);
        case 2: return tanhf(x);
        default: return sigm(x);
    }
}
KVM_GPT_DEVI float act_grad(int kind, float x) {
    switch (kind) {
        case 0: return x > 0.f ? 1.f : 0.f;
        case 1: return dsilu_f(x);
        case 2: { float t = tanhf(x); return 1.f - t * t; }
        default: { float s = sigm(x); return s * (1.f - s); }
    }
}

// ---- elementwise ------------------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void act_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int kind = a[2], e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    k.need(kind >= 0 && kind <= 3, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) o[i] = act_f(kind, x[i]);
}
KVM_GPT_DEVI void act_bwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int kind = a[3], e0 = a[4], e1 = a[5], acc = a[6];
    k.need(e0 >= 0 && e0 <= e1);
    k.need(kind >= 0 && kind <= 3, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    const float* dy = k.fvec(a[1], e1);
    float* dx = k.fvec(a[2], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) { float v = dy[i] * act_grad(kind, x[i]); dx[i] = acc ? dx[i] + v : v; }
}
KVM_GPT_DEVI void silu_mul_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4];
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* g = k.fvec(a[0], e1);
    const float* u = k.fvec(a[1], e1);
    float* o = k.fvec(a[2], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) o[i] = silu_f(g[i]) * u[i];
}
// dgate = dy * up * silu'(gate); dup = dy * silu(gate)   (shared by the fused matmul_tile_silu_mul_bwd, so both agree bitwise)
KVM_GPT_DEVI void silu_mul_bwd_elem(float g, float u, float dy, float& dg, float& du) {
    dg = dy * u * dsilu_f(g);
    du = dy * silu_f(g);
}
KVM_GPT_DEVI void silu_mul_bwd(const KvmCtx& ctx, const int32_t* a) {
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
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) {
        float vg, vu;
        silu_mul_bwd_elem(g[i], u[i], dy[i], vg, vu);
        dg[i] = acc ? dg[i] + vg : vg;
        du[i] = acc ? du[i] + vu : vu;
    }
}

// ---- row softmax: one warp per row ------------------------------------------------------------------------------------------
KVM_GPT_DEVI void softmax_rows_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[2], r1 = a[3], d = a[4], ldx, ldo;
    float scale = af(a, 5);
    k.need(r0 >= 0 && r0 <= r1 && d > 0);
    if (k.bad) return;
    const float* x = k.mat(a[0], r1, d, &ldx);
    float* o = k.mat(a[1], r1, d, &ldo);
    if (k.bad) return;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int r = r0 + w; r < r1; r += KVM_GPT_THREADS / 32) {
        const float* xr = x + (size_t)r * ldx;
        float* orow = o + (size_t)r * ldo;
        float m = -INFINITY;
        for (int c = lane; c < d; c += 32) m = fmaxf(m, xr[c] * scale);
        m = kvmgpt::wmax(m);
        if (m == -INFINITY) m = 0.f;
        float s = 0.f;
        for (int c = lane; c < d; c += 32) s += expf(xr[c] * scale - m);
        s = kvmgpt::wsum(s);
        for (int c = lane; c < d; c += 32) orow[c] = s > 0.f ? expf(xr[c] * scale - m) / s : 0.f;
    }
}
KVM_GPT_DEVI void softmax_rows_bwd(const KvmCtx& ctx, const int32_t* a) {
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
    for (int r = r0 + w; r < r1; r += KVM_GPT_THREADS / 32) {
        const float* pr = p + (size_t)r * ldp;
        const float* dr = dp + (size_t)r * ldq;
        float* xr = dx + (size_t)r * ldx;
        float s = 0.f;
        for (int c = lane; c < d; c += 32) s += pr[c] * dr[c];
        s = kvmgpt::wsum(s);
        for (int c = lane; c < d; c += 32) { float v = scale * pr[c] * (dr[c] - s); xr[c] = acc ? xr[c] + v : v; }
    }
}

// ---- gather / scatter-add -----------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void gather_rows_scaled(const KvmCtx& ctx, const int32_t* a) {
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
    for (int e = threadIdx.x; e < (r1 - r0) * d; e += KVM_GPT_THREADS) {
        int r = r0 + e / d, c = e % d, j = idx[i0 + r];
        float v = 0.f;
        if ((unsigned)j < (unsigned)R) { float s = sc ? sc[i0 + r] : 1.f; v = s * src[(size_t)j * lds + c]; }
        float* o = dst + (size_t)r * ldd + c;
        *o = acc ? *o + v : v;
    }
}
KVM_GPT_DEVI void scatter_add_rows(const KvmCtx& ctx, const int32_t* a) {
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
    for (int e = threadIdx.x; e < (v1 - v0) * d; e += KVM_GPT_THREADS) {
        int v = v0 + e / d, c = e % d;
        float s = 0.f;
        for (int r = 0; r < N; r++)
            if (idx[i0 + r] == v) s += (sc ? sc[i0 + r] : 1.f) * src[(size_t)r * lds + c];
        float* o = dst + (size_t)v * ldd + c;
        *o = acc ? *o + s : s;
    }
}
KVM_GPT_DEVI void row_dot(const KvmCtx& ctx, const int32_t* a) {
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
    for (int r = r0 + w; r < r1; r += KVM_GPT_THREADS / 32) {
        int j = idx ? idx[i0 + r] : r;
        float s = 0.f;
        if ((unsigned)j < (unsigned)R) {
            const float* ar = A + (size_t)j * lda;
            const float* br = B + (size_t)r * ldb;
            for (int c = lane; c < d; c += 32) s += ar[c] * br[c];
        }
        s = kvmgpt::wsum(s);
        if (lane == 0) { float* o = out + (size_t)(i0 + r) * ldo; *o = acc ? *o + s : s; }
    }
}

// ---- mixture-of-experts routing (single task over all tokens) -----------------------------------------------------------------
KVM_GPT_DEVI bool moe_dims(Chk& k, int N, int E, int k_) {
    k.need(N >= 1);
    k.need(E >= 1 && E <= 64 && k_ >= 1 && k_ <= 8 && k_ <= E, KVM_GPT_E_SHAPE);
    return !k.bad;
}
// number of float/int elements of a buffer when used as a flat vector (rows*cols); the buffer must be dense.
KVM_GPT_DEVI long flat_n(Chk& k, int idx, int dtype) {
    const KvmBuf* b = k.get(idx, dtype, false);
    if (!b) return 0;
    if (b->rows > 1 && b->ld != b->cols) { k.fail(KVM_GPT_E_RANGE); return 0; }
    return (long)b->rows * b->cols;
}

KVM_GPT_DEVI void moe_route(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[4], E = a[5], K = a[6], C = a[7], ldp;
    if (!moe_dims(k, N, E, K)) return;
    k.need(C >= 0);
    if (k.bad) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    int32_t* sel = k.ivec(a[1], (long)N * K);
    int32_t* slot = k.ivec(a[2], (long)N * K);
    int32_t* eidx = k.ivec(a[3], (long)E * C);
    long neidx = flat_n(k, a[3], KVM_GPT_I32);
    if (k.bad) return;
    // 1. per token top-k (descending, ties -> lowest expert id): repeated selection of the first maximum among unselected experts
    for (int t = threadIdx.x; t < N; t += KVM_GPT_THREADS) {
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
    for (long i = threadIdx.x; i < neidx; i += KVM_GPT_THREADS) eidx[i] = -1;
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
KVM_GPT_DEVI bool moe_p(const KvmCtx& ctx, const float* probs, int ldp, const int32_t* sel, int t, int K, int E, float* p) {
    for (int j = 0; j < K; j++) {
        int e = sel[(size_t)t * K + j];
        if ((unsigned)e >= (unsigned)E) { ctx.fail(KVM_GPT_E_RANGE); return false; }
        p[j] = probs[(size_t)t * ldp + e];
    }
    return true;
}

KVM_GPT_DEVI void moe_gates_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[4], E = a[5], K = a[6], renorm = a[7], ldp;
    if (!moe_dims(k, N, E, K)) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    const int32_t* slot = k.ivec(a[2], (long)N * K);
    long ng = flat_n(k, a[3], KVM_GPT_F32);
    if (k.bad) return;
    float* gs = (float*)ctx.bufs[a[3]].p;
    for (long i = threadIdx.x; i < ng; i += KVM_GPT_THREADS) gs[i] = 0.f;
    __syncthreads();
    for (int t = threadIdx.x; t < N; t += KVM_GPT_THREADS) {
        float p[8];
        if (!moe_p(ctx, probs, ldp, sel, t, K, E, p)) continue;
        float Z = 1.f;
        if (renorm) { Z = 0.f; for (int j = 0; j < K; j++) Z += p[j]; }
        for (int j = 0; j < K; j++) {
            int s = slot[(size_t)t * K + j];
            if (s < 0) continue;
            if (s >= ng) { ctx.fail(KVM_GPT_E_RANGE); continue; }
            gs[s] = p[j] / Z;
        }
    }
}
KVM_GPT_DEVI void moe_gates_bwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[5], E = a[6], K = a[7], renorm = a[8], acc = a[9], ldp, ldd;
    if (!moe_dims(k, N, E, K)) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    const int32_t* slot = k.ivec(a[2], (long)N * K);
    long ng = flat_n(k, a[3], KVM_GPT_F32);
    float* dprobs = k.mat(a[4], N, E, &ldd);
    if (k.bad) return;
    const float* dgs = (const float*)ctx.bufs[a[3]].p;
    if (!acc) {
        for (int i = threadIdx.x; i < N * E; i += KVM_GPT_THREADS) dprobs[(size_t)(i / E) * ldd + i % E] = 0.f;
        __syncthreads();
    }
    for (int t = threadIdx.x; t < N; t += KVM_GPT_THREADS) {
        float p[8], dg[8];
        if (!moe_p(ctx, probs, ldp, sel, t, K, E, p)) continue;
        bool ok = true;
        for (int j = 0; j < K; j++) {
            int s = slot[(size_t)t * K + j];
            if (s >= ng) { ctx.fail(KVM_GPT_E_RANGE); ok = false; }
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
KVM_GPT_DEVI void moe_counts(const KvmCtx& ctx, const int32_t* sel, int N, int E, int K, int* cnt) {
    if ((int)threadIdx.x < E) {
        int c = 0;
        for (int i = 0; i < N * K; i++) {
            int e = sel[i];
            if ((unsigned)e >= (unsigned)E) { ctx.fail(KVM_GPT_E_RANGE); break; }
            c += e == (int)threadIdx.x;
        }
        cnt[threadIdx.x] = c;
    }
}
KVM_GPT_DEVI void moe_balance_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[3], E = a[4], K = a[5], ldp;
    float coef = af(a, 6);
    if (!moe_dims(k, N, E, K)) return;
    const float* probs = k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    float* loss = k.fvec(a[2], 1);
    if (k.bad) return;
    int* cnt = reinterpret_cast<int*>(kvm_gpt_smem);
    double* fp = reinterpret_cast<double*>(kvm_gpt_smem + 256);
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
KVM_GPT_DEVI void moe_balance_bwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int N = a[3], E = a[4], K = a[5], acc = a[7], ldp, ldd;
    float coef = af(a, 6);
    if (!moe_dims(k, N, E, K)) return;
    k.mat(a[0], N, E, &ldp);
    const int32_t* sel = k.ivec(a[1], (long)N * K);
    float* dprobs = k.mat(a[2], N, E, &ldd);
    if (k.bad) return;
    int* cnt = reinterpret_cast<int*>(kvm_gpt_smem);
    float* g = reinterpret_cast<float*>(kvm_gpt_smem + 256);
    moe_counts(ctx, sel, N, E, K, cnt);
    if ((int)threadIdx.x < E) g[threadIdx.x] = (float)((double)coef * E * ((double)cnt[threadIdx.x] / ((double)N * K)) / (double)N);
    __syncthreads();
    for (int i = threadIdx.x; i < N * E; i += KVM_GPT_THREADS) {
        float* o = dprobs + (size_t)(i / E) * ldd + i % E;
        float v = g[i % E];
        *o = acc ? *o + v : v;
    }
}

// ---- bidirectional attention (kvmgpt tile helpers, causal mask removed) --------------------------------------------------------
template <int HD>
KVM_GPT_DEVI void attn_full_fwd_t(const kvmgpt::AttnArgs& A) {
    if constexpr (HD > KVM_GPT_MAX_HD) return;
    else {
    using T = kvmgpt::AT<HD>;
    float* Qs = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Ks = Qs + T::TILE; float* Vs = Ks + T::TILE; float* Ps = Vs + T::TILE;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, kvh = A.h / (A.H / A.Hkv), Tn = A.T;
    const float* qb = A.q + (size_t)A.b * Tn * A.ldq + A.h * HD;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    kvmgpt::tile_to_smem<HD>(qb, A.ldq, A.i0, Tn, Qs);
    const int gq = A.i0 + ri;
    float m = -INFINITY, l = 0.f, acc[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) acc[c] = 0.f;
    for (int kb = 0, kend = (Tn - 1) / 32; kb <= kend; kb++) {
        __syncthreads();
        kvmgpt::tile_to_smem<HD>(kb_, A.ldk, kb * 32, Tn, Ks);
        kvmgpt::tile_to_smem<HD>(vb_, A.ldv, kb * 32, Tn, Vs);
        __syncthreads();
        float s[4], rm = -INFINITY;
        kvmgpt::dot32<HD>(Qs, Ks, ri, cj, s);
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
        kvmgpt::acc32<HD, false>(acc, Ps, Vs, ri, cj);
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
KVM_GPT_DEVI void attn_full_dq_t(const kvmgpt::AttnArgs& A) {
    if constexpr (HD > KVM_GPT_MAX_HD) return;
    else {
    using T = kvmgpt::AT<HD>;
    float* Qs = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Os = Qs + T::TILE; float* Ks = Os + T::TILE; float* Vs = Ks + T::TILE; float* Ds = Vs + T::TILE;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, kvh = A.h / (A.H / A.Hkv), Tn = A.T, gq = A.i0 + ri;
    const float* kb_ = A.k + (size_t)A.b * Tn * A.ldk + kvh * HD;
    const float* vb_ = A.v + (size_t)A.b * Tn * A.ldv + kvh * HD;
    kvmgpt::tile_to_smem<HD>(A.q + (size_t)A.b * Tn * A.ldq + A.h * HD, A.ldq, A.i0, Tn, Qs);
    kvmgpt::tile_to_smem<HD>(A.dout + (size_t)A.b * Tn * A.ldo + A.h * HD, A.ldo, A.i0, Tn, Os);
    const size_t hrow = ((size_t)A.b * A.H + A.h) * Tn;
    const float lr = gq < Tn ? A.lse[hrow + gq] : 0.f, dr = gq < Tn ? A.delta[hrow + gq] : 0.f;
    float acc[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) acc[c] = 0.f;
    for (int kb = 0, kend = (Tn - 1) / 32; kb <= kend; kb++) {
        __syncthreads();
        kvmgpt::tile_to_smem<HD>(kb_, A.ldk, kb * 32, Tn, Ks);
        kvmgpt::tile_to_smem<HD>(vb_, A.ldv, kb * 32, Tn, Vs);
        __syncthreads();
        float s[4], dp[4];
        kvmgpt::dot32<HD>(Qs, Ks, ri, cj, s);
        kvmgpt::dot32<HD>(Os, Vs, ri, cj, dp);
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
            bool ok = gq < Tn && kb * 32 + cj + 8 * jj < Tn;
            Ds[ri * T::PS + cj + 8 * jj] = ok ? expf(s[jj] * A.scale - lr) * (dp[jj] - dr) : 0.f;
        }
        __syncthreads();
        kvmgpt::acc32<HD, false>(acc, Ds, Ks, ri, cj);
    }
    if (gq < Tn) {
        float* o = A.dq + ((size_t)A.b * Tn + gq) * A.lddq + A.h * HD + cj * T::C;
#pragma unroll
        for (int c = 0; c < T::C; c++) o[c] = acc[c] * A.scale;
    }
    }
}
template <int HD>
KVM_GPT_DEVI void attn_full_dkv_t(const kvmgpt::AttnArgs& A) {
    if constexpr (HD > KVM_GPT_MAX_HD) return;
    else {
    using T = kvmgpt::AT<HD>;
    float* Ks = reinterpret_cast<float*>(kvm_gpt_smem);
    float* Vs = Ks + T::TILE; float* Qs = Vs + T::TILE; float* Os = Qs + T::TILE; float* Ps = Os + T::TILE; float* Ds = Ps + 32 * T::PS;
    const int ri = threadIdx.x >> 3, cj = threadIdx.x & 7, Tn = A.T, G = A.H / A.Hkv, kvh = A.h, k0 = A.i0;
    kvmgpt::tile_to_smem<HD>(A.k + (size_t)A.b * Tn * A.ldk + kvh * HD, A.ldk, k0, Tn, Ks);
    kvmgpt::tile_to_smem<HD>(A.v + (size_t)A.b * Tn * A.ldv + kvh * HD, A.ldv, k0, Tn, Vs);
    float dk[T::C], dv[T::C];
#pragma unroll
    for (int c = 0; c < T::C; c++) { dk[c] = 0.f; dv[c] = 0.f; }
    for (int h = kvh * G; h < (kvh + 1) * G; h++) {
        const float* qb = A.q + (size_t)A.b * Tn * A.ldq + h * HD;
        const float* ob = A.dout + (size_t)A.b * Tn * A.ldo + h * HD;
        const size_t hrow = ((size_t)A.b * A.H + h) * Tn;
        for (int q0 = 0; q0 < Tn; q0 += 32) {
            __syncthreads();
            kvmgpt::tile_to_smem<HD>(qb, A.ldq, q0, Tn, Qs);
            kvmgpt::tile_to_smem<HD>(ob, A.ldo, q0, Tn, Os);
            __syncthreads();
            const int gq = q0 + ri;
            const float lr = gq < Tn ? A.lse[hrow + gq] : 0.f, dr = gq < Tn ? A.delta[hrow + gq] : 0.f;
            float s[4], dp[4];
            kvmgpt::dot32<HD>(Qs, Ks, ri, cj, s);
            kvmgpt::dot32<HD>(Os, Vs, ri, cj, dp);
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
                int gk = k0 + cj + 8 * jj;
                bool ok = gq < Tn && gk < Tn;
                float p = ok ? expf(s[jj] * A.scale - lr) : 0.f;
                Ps[ri * T::PS + cj + 8 * jj] = p;
                Ds[ri * T::PS + cj + 8 * jj] = p * (dp[jj] - dr);
            }
            __syncthreads();
            kvmgpt::acc32<HD, true>(dv, Ps, Os, ri, cj);
            kvmgpt::acc32<HD, true>(dk, Ds, Qs, ri, cj);
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

KVM_GPT_DEVI void attn_full_fwd(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    kvmgpt::AttnArgs A{};
    A.b = a[5]; A.h = a[6]; A.i0 = a[7]; A.T = a[8]; A.H = a[9]; A.Hkv = a[10]; int hd = a[11]; A.scale = af(a, 12);
    if (!kvmgpt::attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.o = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lsew = k.fvec(a[4], (long)(A.b + 1) * A.H * A.T);
    if (k.bad) return;
    k.need(kvmgpt::aligned16(A.q, A.ldq) && kvmgpt::aligned16(A.k, A.ldk) && kvmgpt::aligned16(A.v, A.ldv), KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_full_fwd_t, A)
}
KVM_GPT_DEVI void attn_full_bwd_dq(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    kvmgpt::AttnArgs A{};
    A.b = a[7]; A.h = a[8]; A.i0 = a[9]; A.T = a[10]; A.H = a[11]; A.Hkv = a[12]; int hd = a[13]; A.scale = af(a, 14);
    if (!kvmgpt::attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, false)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl); A.dq = k.mat(a[6], R, (long)A.H * hd, &A.lddq);
    if (k.bad) return;
    k.need(kvmgpt::aligned16(A.q, A.ldq) && kvmgpt::aligned16(A.k, A.ldk) && kvmgpt::aligned16(A.v, A.ldv) && kvmgpt::aligned16(A.dout, A.ldo),
           KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_full_dq_t, A)
}
KVM_GPT_DEVI void attn_full_bwd_dkv(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    kvmgpt::AttnArgs A{};
    A.b = a[8]; A.h = a[9]; A.i0 = a[10]; A.T = a[11]; A.H = a[12]; int hd = a[13]; A.scale = af(a, 14);
    k.need(hd > 0, KVM_GPT_E_SHAPE);
    if (k.bad) return;
    const KvmBuf* kb = k.get(a[1], KVM_GPT_F32, false);   // Hkv = k.cols / hd (no argument slot)
    if (k.bad) return;
    A.Hkv = kb->cols / hd;
    k.need(A.Hkv > 0 && kb->cols % hd == 0, KVM_GPT_E_SHAPE);
    if (k.bad || !kvmgpt::attn_geom(k, A.b, A.h, A.i0, A.T, A.H, A.Hkv, hd, true)) return;
    long R = (long)(A.b + 1) * A.T, nl = R * A.H;
    A.q = k.mat(a[0], R, (long)A.H * hd, &A.ldq); A.k = k.mat(a[1], R, (long)A.Hkv * hd, &A.ldk);
    A.v = k.mat(a[2], R, (long)A.Hkv * hd, &A.ldv); A.dout = k.mat(a[3], R, (long)A.H * hd, &A.ldo);
    A.lse = k.fvec(a[4], nl); A.delta = k.fvec(a[5], nl);
    A.dk = k.mat(a[6], R, (long)A.Hkv * hd, &A.lddk); A.dv = k.mat(a[7], R, (long)A.Hkv * hd, &A.lddv);
    if (k.bad) return;
    k.need(kvmgpt::aligned16(A.q, A.ldq) && kvmgpt::aligned16(A.k, A.ldk) && kvmgpt::aligned16(A.v, A.ldv) && kvmgpt::aligned16(A.dout, A.ldo),
           KVM_GPT_E_SHAPE);
    if (k.bad) return;
    KVM_GPT_HD_SWITCH(hd, attn_full_dkv_t, A)
}

// ---- mixed precision ------------------------------------------------------------------------------------------------------------
// round_bf16 doubles as the storage conversion: x and out are each fp32 or raw bf16 (dtype 2), flat. fp32 -> fp32: rounded values in an
// fp32 container (the emulation); fp32 -> bf16: narrow; bf16 -> fp32: widen (exact); bf16 -> bf16: copy. Rounding is RNE, once per store.
KVM_GPT_DEVI void round_bf16(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[2], e1 = a[3];
    bool xb = false, ob = false;
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const void* x = k.vec_s(a[0], e1, &xb);
    void* o = k.vec_s(a[1], e1, &ob);
    if (k.bad) return;
    if (!xb && !ob) { for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) ((float*)o)[i] = bf16r(((const float*)x)[i]); return; }
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) kvmgpt::st_s(o, i, kvmgpt::ld_s(x, i, xb), ob);
}

// bf16 matmul: kvmgpt::mm_tile<TA, TB, false, true> (64x64 tile, 3-stage cp.async, operands rounded to bf16 when read from shared memory,
// m16n8k16 bf16 tensor-core mma with fp32 accumulate). No precise variant.
KVM_GPT_DEVI void matmul_tile_bf16(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[6], n0 = a[7], M = a[8], N = a[9], K = a[10], fl = a[11];
    bool ta = fl & 1, tb = fl & 2, accum = fl & 4, gelu = fl & 8;
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~31) == 0, KVM_GPT_E_SHAPE);
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
        k.need(((((uintptr_t)A | (uintptr_t)B) & 1) == 0), KVM_GPT_E_SHAPE);
        if (k.bad) return;
    }
    float alpha = af(a, 12);
#define KVM_EX_MM(TA, TB) kvmgpt::mm_tile<TA, TB, false, true>((const float*)A, lda, (const float*)B, ldb, (float*)out, ldo, bias, (const float*)res, ldr, (float*)pre, ldp, m0, n0, M, N, K, alpha, gelu, accum, sd)
    if (!ta && !tb) KVM_EX_MM(false, false); else if (ta && !tb) KVM_EX_MM(true, false);
    else if (!ta && tb) KVM_EX_MM(false, true); else KVM_EX_MM(true, true);
#undef KVM_EX_MM
}

// ---- optimizer schedule / misc ----------------------------------------------------------------------------------------------------
KVM_GPT_DEVI void adamw_lr(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    const float* lr = k.fvec(a[8], 1);
    if (k.bad) return;
    int32_t b[13];
#pragma unroll
    for (int i = 0; i < 13; i++) b[i] = a[i];
    b[8] = __float_as_int(lr[0]);
    kvmgpt::adamw(ctx, b);
}
KVM_GPT_DEVI void lr_schedule(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int kind = a[2], warmup = a[4], total = a[5];
    float base = af(a, 3), mr = af(a, 6);
    k.need(kind >= 0 && kind <= 2, KVM_GPT_E_SHAPE);
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
KVM_GPT_DEVI void scale_flat(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int e0 = a[3], e1 = a[4];
    float alpha = af(a, 2);
    k.need(e0 >= 0 && e0 <= e1);
    if (k.bad) return;
    const float* x = k.fvec(a[0], e1);
    float* o = k.fvec(a[1], e1);
    if (k.bad) return;
    for (int i = e0 + threadIdx.x; i < e1; i += KVM_GPT_THREADS) o[i] = alpha * x[i];
}
KVM_GPT_DEVI void add_rowtile(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int r0 = a[3], r1 = a[4], T = a[5], d = a[6], ldx, ldp, ldo;
    k.need(r0 >= 0 && r0 <= r1 && T > 0 && d > 0);
    if (k.bad) return;
    const float* x = k.mat(a[0], r1, d, &ldx);
    const float* pos = k.mat(a[1], T, d, &ldp);
    float* o = k.mat(a[2], r1, d, &ldo);
    if (k.bad) return;
    for (int e = threadIdx.x; e < (r1 - r0) * d; e += KVM_GPT_THREADS) {
        int r = r0 + e / d, c = e % d;
        o[(size_t)r * ldo + c] = x[(size_t)r * ldx + c] + pos[(size_t)(r % T) * ldp + c];
    }
}

// ---- matmul epilogue wrappers ---------------------------------------------------------------------------------------------------------
// Validate like matmul_tile (flags 4 / 8 rejected), run kvmgpt::matmul_tile into `target`, barrier, then the elementwise second pass
// over the same tile. Each element is read and rewritten by the same thread.
KVM_GPT_DEVI bool mm_wrap_check(Chk& k, int abuf, int bbuf, int tbuf, int m0, int n0, int M, int N, int K, int fl, int32_t* w) {
    bool ta = fl & 1, tb = fl & 2;
    k.need(M > 0 && N > 0 && K > 0 && m0 >= 0 && m0 < M && n0 >= 0 && n0 < N);
    k.need((fl & ~(1 | 2 | 16)) == 0, KVM_GPT_E_SHAPE);
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
KVM_GPT_DEVI void matmul_tile_dgelu(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[4], n0 = a[5], M = a[6], N = a[7], K = a[8], fl = a[9], ldo, ldp;
    int32_t w[13];
    if (!mm_wrap_check(k, a[0], a[1], a[2], m0, n0, M, N, K, fl, w)) return;
    float* out = k.mat(a[2], M, N, &ldo);
    const float* pre = k.mat(a[3], M, N, &ldp);
    if (k.bad) return;
    w[12] = a[10];
    kvmgpt::matmul_tile(ctx, w);
    __syncthreads();
    for (int e = threadIdx.x; e < 64 * 64; e += KVM_GPT_THREADS) {
        int gm = m0 + e / 64, gn = n0 + e % 64;
        if (gm >= M || gn >= N) continue;
        float* o = out + (size_t)gm * ldo + gn;
        *o = *o * kvmgpt::gelu_grad(pre[(size_t)gm * ldp + gn]);
    }
}
KVM_GPT_DEVI void matmul_tile_silu_mul(const KvmCtx& ctx, const int32_t* a) {
    Chk k(ctx);
    int m0 = a[5], n0 = a[6], M = a[7], N = a[8], K = a[9], fl = a[10], ldo, ldg, ldu;
    int32_t w[13];
    if (!mm_wrap_check(k, a[0], a[1], a[4], m0, n0, M, N, K, fl, w)) return;   // matmul result goes to `up`
    float* out = k.mat(a[2], M, N, &ldo);
    const float* gate = k.mat(a[3], M, N, &ldg);
    float* up = k.mat(a[4], M, N, &ldu);
    if (k.bad) return;
    w[12] = a[11];
    kvmgpt::matmul_tile(ctx, w);
    __syncthreads();
    for (int e = threadIdx.x; e < 64 * 64; e += KVM_GPT_THREADS) {
        int gm = m0 + e / 64, gn = n0 + e % 64;
        if (gm >= M || gn >= N) continue;
        out[(size_t)gm * ldo + gn] = silu_f(gate[(size_t)gm * ldg + gn]) * up[(size_t)gm * ldu + gn];
    }
}
KVM_GPT_DEVI void matmul_tile_silu_mul_bwd(const KvmCtx& ctx, const int32_t* a) {
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
    kvmgpt::matmul_tile(ctx, w);
    __syncthreads();
    for (int e = threadIdx.x; e < 64 * 64; e += KVM_GPT_THREADS) {
        int gm = m0 + e / 64, gn = n0 + e % 64;
        if (gm >= M || gn >= N) continue;
        float* pg = dgate + (size_t)lddg * gm + gn;
        float vg, vu;
        silu_mul_bwd_elem(gate[(size_t)gm * ldg + gn], up[(size_t)gm * ldu + gn], *pg, vg, vu);
        *pg = vg;
        dup[(size_t)gm * lddu + gn] = vu;
    }
}

}  // namespace kvmextra

// ---- per-op entry points (same X-macro convention as the gpt pack) -------------------------------------------------------------------
#ifdef KVM_OP
#define X(name, id, sig) KVM_OP(name) { kvmextra::name(c, a); }
#include "kvm_ops_extra.def"
#undef X
#else
#define X(name, id, sig) KVM_GPT_DEVI void kvm_op_##name(const KvmCtx& ctx, const int32_t* args) { kvmextra::name(ctx, args); }
#include "kvm_ops_extra.def"
#undef X
// Returns false for an op id outside the extra pack (ids 96..121).
KVM_GPT_DEVI bool kvm_extra_dispatch(const KvmCtx& ctx, int op, const int32_t* args) {
    switch (op) {
#define X(name, id, sig) case id: kvm_op_##name(ctx, args); return true;
#include "kvm_ops_extra.def"
#undef X
        default: return false;
    }
}
#endif
