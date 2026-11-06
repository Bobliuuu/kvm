#pragma once
// matmul_mma_bf16: the bf16 tensor-core GEMM tile of the KVM op library (A100 sm_80 and newer).
//
//   C[m0:m0+128, n0:n0+256] = alpha * op(A) * op(B)   (+ bias, pre-activation copy, gelu, residual, accumulate: matmul_tile's epilogue)
//
// A and B are raw bf16 storage (KvmBuf.dtype 2); out / res / pre may each be fp32 or bf16 storage; bias is fp32; accumulation is fp32 in
// a fixed order (k16 steps in increasing k, one accumulator per output element), rounded to bf16 once at the store. Shared tiles stay in
// bf16 (no widening): 128 x 256 x 32 per stage, KVM_MMA_NST cp.async stages, 8 warps as 2 (m) x 4 (n) each owning 64 x 64 = 4 m16 x 8 n8
// mma.sync.m16n8k16 fragments, operands fed by ldmatrix (.trans for the m- or n-contiguous orientations, so all four transpose
// combinations run at the same speed). Shared tiles are XOR-swizzled so every ldmatrix and cp.async is bank-conflict free:
//   K-major   [rows][32 k]       : 64 B rows, 16-byte chunk ^= (row >> 1) & 3
//   MN-major  [32 k][128|256 mn] : 256 / 512 B rows, 16-byte chunk ^= row & 7
// Anything the fast path cannot take (fp32 A or B, ld or K not a multiple of 8, a misaligned base) is handed to matmul_tile_bf16 over
// the eight 64 x 64 sub-tiles, so the op accepts every shape and storage mix matmul_tile_bf16 does. Flags: 1 transA, 2 transB, 4 accumulate,
// 8 gelu (no 16/32/...). Separate op id from matmul_tile (fp32/TF32) on purpose: the frontend picks it with the bf16 precision mode.
#include "kvm_ops_gpt.cuh"

#ifndef KVM_MMA_NST
#define KVM_MMA_NST 3          // cp.async pipeline stages (3 fits the executor's 72 KiB of dynamic shared memory with the 128 x 256 tile)
#endif
#define KVM_MMA_BM 128
#define KVM_MMA_BN 256
#define KVM_MMA_BK 32
#define KVM_MMA_SMEM (KVM_MMA_NST * (KVM_MMA_BM + KVM_MMA_BN) * KVM_MMA_BK * 2)

namespace kvmgpt {

KVM_GPT_DEVI void cp_async16_b(unsigned dst, const void* src, int src_bytes) {   // src_bytes 0 zero-fills the 16 bytes and reads nothing
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
KVM_GPT_DEVI void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
KVM_GPT_DEVI void ldsm_x4_t(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// byte offsets inside a stage tile
KVM_GPT_DEVI unsigned mma_off_k(int row, int kc) { return (unsigned)(row * 64 + ((kc ^ ((row >> 1) & 3)) << 4)); }   // K-major, kc = 16-byte chunk (0..3)
KVM_GPT_DEVI unsigned mma_off_mn(int row, int c) { return (unsigned)(row * 256 + ((c ^ (row & 7)) << 4)); }          // MN-major, c = 16-byte chunk (0..15)

// One ROWS-row (K-major) or ROWS-column (MN-major) operand tile of one k slab into shared memory. KMAJ: global is [mn][k] (K contiguous);
// else [k][mn]. Chunks (8 bf16) that start outside [MN, K) are zero-filled by the hardware; the caller guarantees whole-chunk extents.
template <bool KMAJ, int ROWS>
KVM_GPT_DEVI void mma_load_tile(unsigned dst, const uint16_t* src, int ld, int mn0, int k0, int MN, int K) {
    constexpr int CPR = ROWS / 8;   // MN-major chunks per k row
#pragma unroll
    for (int s = 0; s < ROWS * 4 / KVM_GPT_THREADS; s++) {
        const int q = threadIdx.x + s * KVM_GPT_THREADS;
        if (KMAJ) {
            const int row = q >> 2, kc = q & 3, gm = mn0 + row, gk = k0 + kc * 8;
            const bool ok = gm < MN && gk < K;
            cp_async16_b(dst + mma_off_k(row, kc), src + (ok ? (size_t)gm * ld + gk : 0), ok ? 16 : 0);
        } else {
            const int row = q / CPR, c = q % CPR, gk = k0 + row, gn = mn0 + c * 8;
            const bool ok = gk < K && gn < MN;
            cp_async16_b(dst + (unsigned)(row * ROWS * 2 + ((c ^ (row & 7)) << 4)), src + (ok ? (size_t)gk * ld + gn : 0), ok ? 16 : 0);
        }
    }
}

template <bool TA, bool TB>
KVM_GPT_DEVI void mma_bf16_tile(const uint16_t* A, int lda, const uint16_t* B, int ldb, void* out, int ldo, bool ob, const float* bias,
                                const void* res, int ldr, bool rb, void* pre, int ldp, bool pb, int m0, int n0, int M, int N, int K,
                                float alpha, bool gelu, bool accum) {
    constexpr int BM = KVM_MMA_BM, BN = KVM_MMA_BN, BK = KVM_MMA_BK, NST = KVM_MMA_NST, STAGE = (BM + BN) * BK * 2, ATILE = BM * BK * 2;
    constexpr int NJ = 8;   // n8 fragments per warp (warp tile 64 x 64; 8 warps as 2 (m) x 4 (n))
    static_assert(BK == 32 && BM == 128 && BN == 256 && NST >= 3, "tile geometry is fixed");
    const unsigned sm = smem_u32(kvm_gpt_smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wm = warp >> 2, wn = warp & 3, g = lane >> 2, t = lane & 3;
    float acc[4][NJ][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    const int nk = (K + BK - 1) / BK;
    auto issue = [&](int it) {
        const unsigned a_s = sm + (it % NST) * STAGE, b_s = a_s + ATILE;
        if (TA) mma_load_tile<false, BM>(a_s, A, lda, m0, it * BK, M, K); else mma_load_tile<true, BM>(a_s, A, lda, m0, it * BK, M, K);
        if (TB) mma_load_tile<true, BN>(b_s, B, ldb, n0, it * BK, N, K); else mma_load_tile<false, BN>(b_s, B, ldb, n0, it * BK, N, K);
    };
    unsigned af[2][4][4], bf[2][NJ][2];   // fragments of two consecutive k16 steps: one set is loaded while the other feeds the tensor cores
    auto load_frags = [&](int buf, int it, int kk) {
        const unsigned a_s = sm + (it % NST) * STAGE, b_s = a_s + ATILE;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int mb = wm * 64 + i * 16;
            if (!TA) ldsm_x4(af[buf][i], a_s + mma_off_k(mb + (lane & 7) + 8 * ((lane >> 3) & 1), (kk >> 3) + (lane >> 4)));
            else ldsm_x4_t(af[buf][i], a_s + (unsigned)((kk + (lane & 7) + 8 * (lane >> 4)) * (BM * 2) + ((((mb >> 3) + ((lane >> 3) & 1)) ^ (lane & 7)) << 4)));
        }
#pragma unroll
        for (int jp = 0; jp < NJ / 2; jp++) {
            const int nb = wn * 64 + jp * 16;
            unsigned r[4];
            if (TB) ldsm_x4(r, b_s + mma_off_k(nb + (lane & 7) + 8 * (lane >> 4), (kk >> 3) + ((lane >> 3) & 1)));
            else ldsm_x4_t(r, b_s + (unsigned)((kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * (BN * 2) + ((((nb >> 3) + (lane >> 4)) ^ (lane & 7)) << 4)));
            bf[buf][2 * jp][0] = r[0]; bf[buf][2 * jp][1] = r[1]; bf[buf][2 * jp + 1][0] = r[2]; bf[buf][2 * jp + 1][1] = r[3];
        }
    };
    auto mma_step = [&](int buf) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) mma_bf16(acc[i][j], af[buf][i], bf[buf][j][0], bf[buf][j][1]);
    };
#pragma unroll
    for (int p = 0; p < NST - 1; p++) { if (p < nk) issue(p); cp_commit(); }
    cp_wait<NST - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
    for (int it = 0; it < nk; it++) {
        load_frags(1, it, 16);
        mma_step(0);
        cp_wait<NST - 3>();                    // stage it+1 landed (this thread's copies) ...
        __syncthreads();                       // ... for every thread, and nobody still reads the stage refilled next
        if (it + NST - 1 < nk) issue(it + NST - 1);
        cp_commit();
        if (it + 1 < nk) load_frags(0, it + 1, 0);
        mma_step(1);
    }
    cp_wait<0>();
    // Epilogue straight from the accumulators: a thread holds (row g | g+8, columns 2t, 2t+1) of every fragment. Column pairs leave as one
    // 32-bit (bf16x2) or 64-bit (fp32) access when every operand allows it.
    const bool pair = (((uintptr_t)out | (uintptr_t)res | (uintptr_t)pre | (uintptr_t)bias) & 7) == 0 && (ldo & 1) == 0 && (ldr & 1) == 0 && (ldp & 1) == 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {
                const int gm = m0 + wm * 64 + i * 16 + g + hf * 8, gn = n0 + wn * 64 + j * 8 + 2 * t;
                if (gm >= M || gn >= N) continue;
                float v[2] = {acc[i][j][2 * hf] * alpha, acc[i][j][2 * hf + 1] * alpha};
                const int nv = (gn + 1 < N) ? 2 : 1;
                const size_t oi = (size_t)gm * ldo + gn, ri = (size_t)gm * ldr + gn, pi = (size_t)gm * ldp + gn;
                if (pair && nv == 2) {
                    if (bias) { const float2 b2 = *reinterpret_cast<const float2*>(bias + gn); v[0] += b2.x; v[1] += b2.y; }
                    if (pre) {
                        if (pb) *reinterpret_cast<unsigned*>((uint16_t*)pre + pi) = pack_bf16(v[0], v[1]);
                        else *reinterpret_cast<float2*>((float*)pre + pi) = make_float2(v[0], v[1]);
                    }
                    if (gelu) { v[0] = gelu_f(v[0]); v[1] = gelu_f(v[1]); }
                    if (res) {
                        if (rb) { const unsigned r2 = *reinterpret_cast<const unsigned*>((const uint16_t*)res + ri); v[0] += __uint_as_float(r2 << 16); v[1] += __uint_as_float(r2 & 0xffff0000u); }
                        else { const float2 r2 = *reinterpret_cast<const float2*>((const float*)res + ri); v[0] += r2.x; v[1] += r2.y; }
                    }
                    if (ob) {
                        unsigned* o = reinterpret_cast<unsigned*>((uint16_t*)out + oi);
                        if (accum) { const unsigned o2 = *o; v[0] += __uint_as_float(o2 << 16); v[1] += __uint_as_float(o2 & 0xffff0000u); }
                        *o = pack_bf16(v[0], v[1]);
                    } else {
                        float2* o = reinterpret_cast<float2*>((float*)out + oi);
                        if (accum) { const float2 o2 = *o; v[0] += o2.x; v[1] += o2.y; }
                        *o = make_float2(v[0], v[1]);
                    }
                } else {
                    for (int e = 0; e < nv; e++) {
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
}

}  // namespace kvmgpt
