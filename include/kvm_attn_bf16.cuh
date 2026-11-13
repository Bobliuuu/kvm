// Causal flash attention on bf16 operands: mma.sync m16n8k16 (bf16 in, fp32 accumulate), ldmatrix-fed, swizzled bf16 shared tiles.
//
// Included INSIDE namespace kvmgpt by kvm_ops_gpt.cuh (it needs AttnArgs, cp_commit/cp_wait, mma_bf16, pack_bf16, st_pair, ex2f).
// attn_fwd / attn_bwd_dq / attn_bwd_dkv use it when q, k, v (and dO for the backward) are raw bf16 storage, 16-byte aligned with a leading
// dimension a multiple of 8, and the head size is 64. Every product of two bf16 values is exact in fp32; what differs from the 3xTF32
// path is the bf16 rounding of P (forward, dV) and dS (dQ, dK) before they feed the tensor cores, as in every bf16 flash attention.
// Structure per task is that of attn_fwd_t / attn_dq_t / attn_dkv_t: 128 rows per task, one m16 row tile per warp (no cross-warp
// reduction), 64-row K/V (or Q/dO) tiles streamed through a 3-stage cp.async pipeline, the warp's own operand in registers.
//
// Shared tile of 64 rows x 64 bf16 (128-byte rows): row r, 16-byte chunk c at byte r*128 + ((c ^ (r & 7)) << 4): every ldmatrix phase
// (eight rows, one chunk column) hits eight distinct 16-byte bank groups.

#ifndef KVM_GPT_ATTN_BF16
#define KVM_GPT_ATTN_BF16 1   // 1: bf16-storage attention runs on bf16 tensor-core tiles (this file); 0: always the 3xTF32 path (A/B switch)
#endif

namespace ab {
constexpr int HD = 64, KT = 64, CH = HD / 8, NK = HD / 16, ND = HD / 8, TILE = KT * HD * 2, NST = 3;
constexpr int STAGE = 2 * TILE, STAGE_KV = 2 * TILE + 512;   // dkv stages also hold the tile's 64 lse and 64 delta values
static_assert(NST * STAGE_KV <= 56 * 1024, "bf16 attention tiles must fit the launch's dynamic shared memory");
}  // namespace ab

KVM_GPT_DEVI unsigned ab_sw(int r, int c) { return (unsigned)(r * 128 + ((c ^ (r & 7)) << 4)); }
KVM_GPT_DEVI void ab_ldsm(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
KVM_GPT_DEVI void ab_ldsm_t(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
KVM_GPT_DEVI void ab_cp16(unsigned dst, const void* src, int bytes) {   // bytes 0 zero-fills and reads nothing
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
KVM_GPT_DEVI void ab_cp4(unsigned dst, const void* src, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
// 64 rows x 64 of a bf16 matrix (rows row0.., leading dimension ldg, `src` at the head's first column) into a swizzled tile; rows >= rmax zero.
KVM_GPT_DEVI void ab_tile(unsigned dst, const uint16_t* src, int ldg, int row0, int rmax) {
#pragma unroll
    for (int i = 0; i < ab::KT * ab::CH / KVM_GPT_THREADS; i++) {
        const int q = threadIdx.x + i * KVM_GPT_THREADS, r = q / ab::CH, c = q % ab::CH, gr = row0 + r;
        ab_cp16(dst + ab_sw(r, c), gr < rmax ? src + (size_t)gr * ldg + c * 8 : src, gr < rmax ? 16 : 0);
    }
}
// A fragments (m16 x k16 per step) of the warp's 16 rows: row g / g+8, columns ks*16 + 2t(+1) and +8; rows >= rmax are zero.
KVM_GPT_DEVI void ab_load_a(unsigned (&a)[ab::NK][4], const uint16_t* base, int ld, int row, int rmax, int t) {
    const bool v0 = row < rmax, v1 = row + 8 < rmax;
    const uint16_t* p0 = base + (size_t)row * ld + 2 * t;
    const uint16_t* p1 = p0 + (size_t)8 * ld;
#pragma unroll
    for (int ks = 0; ks < ab::NK; ks++) {
        a[ks][0] = v0 ? *reinterpret_cast<const unsigned*>(p0 + ks * 16) : 0u;      a[ks][1] = v1 ? *reinterpret_cast<const unsigned*>(p1 + ks * 16) : 0u;
        a[ks][2] = v0 ? *reinterpret_cast<const unsigned*>(p0 + ks * 16 + 8) : 0u;  a[ks][3] = v1 ? *reinterpret_cast<const unsigned*>(p1 + ks * 16 + 8) : 0u;
    }
}
// acc[8][4] (16 x 64) += X (the warp's A fragments, k = hd) * T^T, T a 64-row tile read as B (n = tile row, k = hd): S = Q K^T and friends.
KVM_GPT_DEVI void ab_xt(float (&acc)[8][4], const unsigned (&a)[ab::NK][4], unsigned tile, int lane) {
#pragma unroll
    for (int ks = 0; ks < ab::NK; ks++)
#pragma unroll
        for (int jp = 0; jp < 4; jp++) {
            unsigned r[4];
            ab_ldsm(r, tile + ab_sw(jp * 16 + (lane & 7) + 8 * (lane >> 4), ks * 2 + ((lane >> 3) & 1)));
            mma_bf16(acc[2 * jp], a[ks], r[0], r[1]);
            mma_bf16(acc[2 * jp + 1], a[ks], r[2], r[3]);
        }
}
// o[8][4] (16 x 64) += P * T, P the 16 x 64 probability-like tile in the accumulator registers p (rounded to bf16 here), T a 64-row tile
// read as B (k = tile row, n = hd) through ldmatrix.trans.
KVM_GPT_DEVI void ab_pt(float (&o)[8][4], const float (&p)[8][4], unsigned tile, int lane) {
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        unsigned pa[4] = {pack_bf16(p[2 * kk][0], p[2 * kk][1]), pack_bf16(p[2 * kk][2], p[2 * kk][3]),
                          pack_bf16(p[2 * kk + 1][0], p[2 * kk + 1][1]), pack_bf16(p[2 * kk + 1][2], p[2 * kk + 1][3])};
#pragma unroll
        for (int dp = 0; dp < ab::ND / 2; dp++) {
            unsigned r[4];
            ab_ldsm_t(r, tile + ab_sw(kk * 16 + (lane & 7) + 8 * ((lane >> 3) & 1), dp * 2 + (lane >> 4)));
            mma_bf16(o[2 * dp], pa, r[0], r[1]);
            mma_bf16(o[2 * dp + 1], pa, r[2], r[3]);
        }
    }
}

KVM_GPT_DEVI void attn_fwd_b(const AttnArgs& A) {
    const unsigned sm = smem_u32(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kvh = A.h / (A.H / A.Hkv), Tn = A.T, wrow = A.i0 + warp * 16, rowA = wrow + g, rowB = rowA + 8;
    const uint16_t* kb = reinterpret_cast<const uint16_t*>(A.k) + (size_t)A.b * Tn * A.ldk + kvh * ab::HD;
    const uint16_t* vb = reinterpret_cast<const uint16_t*>(A.v) + (size_t)A.b * Tn * A.ldv + kvh * ab::HD;
    const int nk = (min(A.i0 + KVM_GPT_ATTN_ROWS, Tn) + ab::KT - 1) / ab::KT;
    auto issue = [&](int it) {
        if (it < nk) {
            const unsigned st = sm + (it % ab::NST) * ab::STAGE;
            ab_tile(st, kb, A.ldk, it * ab::KT, Tn);
            ab_tile(st + ab::TILE, vb, A.ldv, it * ab::KT, Tn);
        }
        cp_commit();   // empty groups too, so the wait counts stay uniform
    };
    issue(0); issue(1);
    unsigned aq[ab::NK][4];
    ab_load_a(aq, reinterpret_cast<const uint16_t*>(A.q) + (size_t)A.b * Tn * A.ldq + A.h * ab::HD, A.ldq, rowA, Tn, t);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, o[ab::ND][4];
#pragma unroll
    for (int n = 0; n < ab::ND; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[n][e] = 0.f;
    const float c2 = A.scale * KVM_GPT_L2E;
    for (int it = 0; it < nk; it++) {
        cp_wait<1>();
        __syncthreads();   // tile `it` landed; every warp finished tile it-1, so the stage the next issue fills is free
        issue(it + 2);
        const int kbase = it * ab::KT;
        if (kbase > wrow + 15 || wrow >= Tn) continue;   // all of this tile's keys lie above all of the warp's rows (warp-uniform)
        const unsigned Ks = sm + (it % ab::NST) * ab::STAGE, Vs = Ks + ab::TILE;
        float s[8][4];
#pragma unroll
        for (int kt = 0; kt < 8; kt++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[kt][e] = 0.f;
        ab_xt(s, aq, Ks, lane);
        float r0m = -INFINITY, r1m = -INFINITY;
#pragma unroll
        for (int kt = 0; kt < 8; kt++) {
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
        for (int kt = 0; kt < 8; kt++) {
            s[kt][0] = ex2f(s[kt][0] - ms0); s[kt][1] = ex2f(s[kt][1] - ms0);
            s[kt][2] = ex2f(s[kt][2] - ms1); s[kt][3] = ex2f(s[kt][3] - ms1);
            ps0 += s[kt][0] + s[kt][1]; ps1 += s[kt][2] + s[kt][3];
        }
        l0 = l0 * al0 + ps0; l1 = l1 * al1 + ps1;
#pragma unroll
        for (int n = 0; n < ab::ND; n++) { o[n][0] *= al0; o[n][1] *= al0; o[n][2] *= al1; o[n][3] *= al1; }
        ab_pt(o, s, Vs, lane);
    }
    cp_wait<0>();
    if (wrow >= Tn) return;
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1); l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1); l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = l0 > 0.f ? 1.f / l0 : 0.f, inv1 = l1 > 0.f ? 1.f / l1 : 0.f;
    const bool obf = A.sd & KVM_ATT_O;
    const size_t oa = ((size_t)A.b * Tn + rowA) * A.ldo + A.h * ab::HD + 2 * t, ob = oa + (size_t)8 * A.ldo;
#pragma unroll
    for (int n = 0; n < ab::ND; n++) {
        if (rowA < Tn) st_pair(A.o, oa + n * 8, o[n][0] * inv0, o[n][1] * inv0, obf);
        if (rowB < Tn) st_pair(A.o, ob + n * 8, o[n][2] * inv1, o[n][3] * inv1, obf);
    }
    if (t == 0) {
        float* ls = A.lsew + ((size_t)A.b * A.H + A.h) * Tn;
        if (rowA < Tn) ls[rowA] = l0 > 0.f ? (m0 + log2f(l0)) * KVM_GPT_LN2 : 0.f;
        if (rowB < Tn) ls[rowB] = l1 > 0.f ? (m1 + log2f(l1)) * KVM_GPT_LN2 : 0.f;
    }
}

// dQ = scale * dS K, dS = P (dP - delta), P = exp(scale*QK^T - lse), dP = dO V^T (P recomputed from lse).
KVM_GPT_DEVI void attn_dq_b(const AttnArgs& A) {
    const unsigned sm = smem_u32(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int kvh = A.h / (A.H / A.Hkv), Tn = A.T, wrow = A.i0 + warp * 16, rowA = wrow + g, rowB = rowA + 8;
    const uint16_t* kb = reinterpret_cast<const uint16_t*>(A.k) + (size_t)A.b * Tn * A.ldk + kvh * ab::HD;
    const uint16_t* vb = reinterpret_cast<const uint16_t*>(A.v) + (size_t)A.b * Tn * A.ldv + kvh * ab::HD;
    const int nk = (min(A.i0 + KVM_GPT_ATTN_ROWS, Tn) + ab::KT - 1) / ab::KT;
    auto issue = [&](int it) {
        if (it < nk) {
            const unsigned st = sm + (it % ab::NST) * ab::STAGE;
            ab_tile(st, kb, A.ldk, it * ab::KT, Tn);
            ab_tile(st + ab::TILE, vb, A.ldv, it * ab::KT, Tn);
        }
        cp_commit();
    };
    issue(0); issue(1);
    unsigned aq[ab::NK][4], ad[ab::NK][4];
    ab_load_a(aq, reinterpret_cast<const uint16_t*>(A.q) + (size_t)A.b * Tn * A.ldq + A.h * ab::HD, A.ldq, rowA, Tn, t);
    ab_load_a(ad, reinterpret_cast<const uint16_t*>(A.dout) + (size_t)A.b * Tn * A.ldo + A.h * ab::HD, A.ldo, rowA, Tn, t);
    const size_t hrow = ((size_t)A.b * A.H + A.h) * Tn;
    const float lr0 = rowA < Tn ? A.lse[hrow + rowA] * KVM_GPT_L2E : 0.f, lr1 = rowB < Tn ? A.lse[hrow + rowB] * KVM_GPT_L2E : 0.f;
    const float dr0 = rowA < Tn ? A.delta[hrow + rowA] : 0.f, dr1 = rowB < Tn ? A.delta[hrow + rowB] : 0.f;
    float acc[ab::ND][4];
#pragma unroll
    for (int n = 0; n < ab::ND; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) acc[n][e] = 0.f;
    const float c2 = A.scale * KVM_GPT_L2E;
    for (int it = 0; it < nk; it++) {
        cp_wait<1>();
        __syncthreads();
        issue(it + 2);
        const int kbase = it * ab::KT;
        if (kbase > wrow + 15 || wrow >= Tn) continue;
        const unsigned Ks = sm + (it % ab::NST) * ab::STAGE, Vs = Ks + ab::TILE;
        float s[8][4], dp[8][4];
#pragma unroll
        for (int kt = 0; kt < 8; kt++)
#pragma unroll
            for (int e = 0; e < 4; e++) { s[kt][e] = 0.f; dp[kt][e] = 0.f; }
        ab_xt(s, aq, Ks, lane);
        ab_xt(dp, ad, Vs, lane);
#pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            const int key = kbase + kt * 8 + 2 * t;
            s[kt][0] = p_of(s[kt][0], c2, lr0, key <= rowA) * (dp[kt][0] - dr0);
            s[kt][1] = p_of(s[kt][1], c2, lr0, key + 1 <= rowA) * (dp[kt][1] - dr0);
            s[kt][2] = p_of(s[kt][2], c2, lr1, key <= rowB) * (dp[kt][2] - dr1);
            s[kt][3] = p_of(s[kt][3], c2, lr1, key + 1 <= rowB) * (dp[kt][3] - dr1);
        }
        ab_pt(acc, s, Ks, lane);
    }
    cp_wait<0>();
    if (wrow >= Tn) return;
    const bool qbf = A.sd & KVM_ATT_DQ;
    const size_t oa = ((size_t)A.b * Tn + rowA) * A.lddq + A.h * ab::HD + 2 * t, ob = oa + (size_t)8 * A.lddq;
#pragma unroll
    for (int n = 0; n < ab::ND; n++) {
        if (rowA < Tn) st_pair(A.dq, oa + n * 8, acc[n][0] * A.scale, acc[n][1] * A.scale, qbf);
        if (rowB < Tn) st_pair(A.dq, ob + n * 8, acc[n][2] * A.scale, acc[n][3] * A.scale, qbf);
    }
}

// dV = sum P^T dO, dK = scale * sum dS^T Q over the GQA group's heads and all queries >= the key block. The warp's 16 keys stay resident
// as K and V fragments; (head, 64 queries) tiles of Q and dO stream through the pipeline together with their lse and delta values.
// S^T = K Q^T has the keys as mma rows, so P^T and dS^T leave the accumulators already in the A layout of dV += P^T dO, dK += dS^T Q.
KVM_GPT_DEVI void attn_dkv_b(const AttnArgs& A) {
    const unsigned sm = smem_u32(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int Tn = A.T, G = A.H / A.Hkv, kvh = A.h, k0 = A.i0, wkey = k0 + warp * 16, keyA = wkey + g, keyB = keyA + 8;
    const int nq = (Tn - k0 + ab::KT - 1) / ab::KT, nit = G * nq;
    auto issue = [&](int it) {
        if (it < nit) {
            const int h = kvh * G + it / nq, q0 = k0 + (it % nq) * ab::KT;
            const unsigned st = sm + (it % ab::NST) * ab::STAGE_KV;
            ab_tile(st, reinterpret_cast<const uint16_t*>(A.q) + (size_t)A.b * Tn * A.ldq + h * ab::HD, A.ldq, q0, Tn);
            ab_tile(st + ab::TILE, reinterpret_cast<const uint16_t*>(A.dout) + (size_t)A.b * Tn * A.ldo + h * ab::HD, A.ldo, q0, Tn);
            if (threadIdx.x < 128) {   // lse (threads 0-63) and delta (64-127) of the tile's queries, zero beyond T
                const int q = q0 + (threadIdx.x & 63);
                const float* src = (threadIdx.x < 64 ? A.lse : A.delta) + ((size_t)A.b * A.H + h) * Tn;
                ab_cp4(st + 2 * ab::TILE + threadIdx.x * 4, q < Tn ? src + q : src, q < Tn ? 4 : 0);
            }
        }
        cp_commit();
    };
    issue(0); issue(1);
    unsigned ak[ab::NK][4], av[ab::NK][4];
    ab_load_a(ak, reinterpret_cast<const uint16_t*>(A.k) + (size_t)A.b * Tn * A.ldk + kvh * ab::HD, A.ldk, keyA, Tn, t);
    ab_load_a(av, reinterpret_cast<const uint16_t*>(A.v) + (size_t)A.b * Tn * A.ldv + kvh * ab::HD, A.ldv, keyA, Tn, t);
    float dk[ab::ND][4], dv[ab::ND][4];
#pragma unroll
    for (int n = 0; n < ab::ND; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) { dk[n][e] = 0.f; dv[n][e] = 0.f; }
    const float c2 = A.scale * KVM_GPT_L2E;
    for (int it = 0; it < nit; it++) {
        cp_wait<1>();
        __syncthreads();
        issue(it + 2);
        const int q0 = k0 + (it % nq) * ab::KT;
        if (q0 + ab::KT - 1 < wkey || wkey >= Tn) continue;   // every query of this tile precedes every key of this warp (warp-uniform)
        const unsigned Qs = sm + (it % ab::NST) * ab::STAGE_KV, Os = Qs + ab::TILE;
        const float* lsm = reinterpret_cast<const float*>(kvm_gpt_smem + (it % ab::NST) * ab::STAGE_KV + 2 * ab::TILE);
        float s[8][4], dp[8][4];
#pragma unroll
        for (int kt = 0; kt < 8; kt++)
#pragma unroll
            for (int e = 0; e < 4; e++) { s[kt][e] = 0.f; dp[kt][e] = 0.f; }
        ab_xt(s, ak, Qs, lane);
        ab_xt(dp, av, Os, lane);
        float p[8][4];
#pragma unroll
        for (int kt = 0; kt < 8; kt++) {
            const int ql = kt * 8 + 2 * t, qa = q0 + ql, qb = qa + 1;
            const float la = lsm[ql] * KVM_GPT_L2E, lb = lsm[ql + 1] * KVM_GPT_L2E, da = lsm[64 + ql], db = lsm[64 + ql + 1];
            p[kt][0] = p_of(s[kt][0], c2, la, qa < Tn && keyA <= qa);   p[kt][1] = p_of(s[kt][1], c2, lb, qb < Tn && keyA <= qb);
            p[kt][2] = p_of(s[kt][2], c2, la, qa < Tn && keyB <= qa);   p[kt][3] = p_of(s[kt][3], c2, lb, qb < Tn && keyB <= qb);
            s[kt][0] = p[kt][0] * (dp[kt][0] - da); s[kt][1] = p[kt][1] * (dp[kt][1] - db);
            s[kt][2] = p[kt][2] * (dp[kt][2] - da); s[kt][3] = p[kt][3] * (dp[kt][3] - db);
        }
        ab_pt(dv, p, Os, lane);
        ab_pt(dk, s, Qs, lane);
    }
    cp_wait<0>();
    if (wkey >= Tn) return;
    const bool kbf = A.sd & KVM_ATT_DK, vbf = A.sd & KVM_ATT_DV;
    const size_t da = ((size_t)A.b * Tn + keyA) * A.lddk + kvh * ab::HD + 2 * t, db = da + (size_t)8 * A.lddk;
    const size_t va = ((size_t)A.b * Tn + keyA) * A.lddv + kvh * ab::HD + 2 * t, vb = va + (size_t)8 * A.lddv;
#pragma unroll
    for (int n = 0; n < ab::ND; n++) {
        if (keyA < Tn) { st_pair(A.dk, da + n * 8, dk[n][0] * A.scale, dk[n][1] * A.scale, kbf); st_pair(A.dv, va + n * 8, dv[n][0], dv[n][1], vbf); }
        if (keyB < Tn) { st_pair(A.dk, db + n * 8, dk[n][2] * A.scale, dk[n][3] * A.scale, kbf); st_pair(A.dv, vb + n * 8, dv[n][2], dv[n][3], vbf); }
    }
}

// Whether a task takes the bf16 tensor-core path: head size 64, every operand the kernel reads as bf16 storage, 16-byte aligned, ld % 8 == 0.
KVM_GPT_DEVI bool ab_ok(const AttnArgs& A, int hd, bool bwd) {
    if (!KVM_GPT_ATTN_BF16 || hd != ab::HD) return false;
    const unsigned need = KVM_ATT_Q | KVM_ATT_K | KVM_ATT_V | (bwd ? KVM_ATT_DO : 0);
    if ((A.sd & need) != need) return false;
    const uintptr_t pa = (uintptr_t)A.q | (uintptr_t)A.k | (uintptr_t)A.v | (bwd ? (uintptr_t)A.dout : 0);
    return (pa & 15) == 0 && ((A.ldq | A.ldk | A.ldv | (bwd ? A.ldo : 0)) & 7) == 0;
}
