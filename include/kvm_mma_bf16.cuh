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
#define KVM_MMA_NST 4          // cp.async pipeline stages (4 x 24 KiB; 3 stages fit 72 KiB but run 7% slower at 4096^3)
#endif
#define KVM_MMA_BM 128
#define KVM_MMA_BN 256
#define KVM_MMA_BK 32
#define KVM_MMA_SMEM (KVM_MMA_NST * (KVM_MMA_BM + KVM_MMA_BN) * KVM_MMA_BK * 2)
static_assert(KVM_MMA_SMEM <= kvm_gpt_smem_bytes(0), "matmul_mma_bf16's stage buffers must fit the dynamic shared memory the executor launches with");

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

// Global -> shared copy of one operand tile (ROWS rows of K-major data, or ROWS columns of MN-major data) of one k slab, as ROWS*4 16-byte
// chunks spread over the 256 threads (ROWS*4/256 per thread). KMAJ: global is [mn][k] (K contiguous); else [k][mn]. A chunk (8 bf16) that
// starts outside [MN, K) is zero-filled by the hardware (the caller guarantees whole-chunk extents). Addresses are set up once; issuing a chunk
// costs one 64-bit add and two compares. (Spreading the copies between the tensor-core instructions of a k step, and a plain cp.async form for
// interior tiles, were both measured and were not faster: results/2026-11-06-a100-bf16.)
template <bool KMAJ, int ROWS>
struct MmaLoader {
    static constexpr int NS = ROWS * 4 / KVM_GPT_THREADS;                          // chunks per thread
    static constexpr int CPR = ROWS / 8;                                           // MN-major chunks per k row
    static constexpr int RSTEP = KMAJ ? KVM_GPT_THREADS / 4 : KVM_GPT_THREADS / CPR;   // rows between a thread's consecutive chunks
    const uint16_t* base;   // chunk 0, k slab 0 (a valid address even when the chunk is out of range: zero-filled copies read nothing)
    size_t sstride, kstep;  // elements between consecutive chunks of a thread / between k slabs
    unsigned dst;           // byte offset of chunk 0 inside the tile (consecutive chunks are 4096 bytes apart for both layouts)
    int mn, k, MN, K;       // chunk 0: first mn index (row or column) and k index it covers; extents
    KVM_GPT_DEVI void init(const uint16_t* src, int ld, int mn0, int MN_, int K_) {
        const int tid = threadIdx.x;
        MN = MN_; K = K_;
        if (KMAJ) {
            const int row = tid >> 2, kc = tid & 3;
            mn = mn0 + row; k = kc * 8;
            base = src + (mn < MN ? (size_t)mn * ld + k : 0);
            kstep = KVM_MMA_BK; dst = mma_off_k(row, kc);
        } else {
            const int row = tid / CPR, c = tid % CPR;
            mn = mn0 + c * 8; k = row;
            base = src + (mn < MN ? (size_t)row * ld + mn : 0);
            kstep = (size_t)KVM_MMA_BK * ld; dst = (unsigned)(row * ROWS * 2 + ((c ^ (row & 7)) << 4));
        }
        sstride = (size_t)RSTEP * ld;
    }
    // chunk s of k slab `it` into the tile at shared address `tile`
    KVM_GPT_DEVI void issue(unsigned tile, int it, int s) const {
        const bool ok = KMAJ ? (mn + s * RSTEP < MN && k + it * KVM_MMA_BK < K) : (mn < MN && k + s * RSTEP + it * KVM_MMA_BK < K);
        cp_async16_b(tile + dst + s * 4096, ok ? base + (size_t)it * kstep + (size_t)s * sstride : base, ok ? 16 : 0);
    }
};

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
    MmaLoader<!TA, BM> la;
    MmaLoader<TB, BN> lb;
    la.init(A, lda, m0, M, K);
    lb.init(B, ldb, n0, N, K);
    constexpr int NCH = MmaLoader<!TA, BM>::NS + MmaLoader<TB, BN>::NS;   // 6 chunks per thread and k slab
    auto issue_chunk = [&](int it, int c) {   // chunk c (0..5) of slab it into its stage
        const unsigned a_s = sm + (it % NST) * STAGE;
        if (c < MmaLoader<!TA, BM>::NS) la.issue(a_s, it, c); else lb.issue(a_s + ATILE, it, c - MmaLoader<!TA, BM>::NS);
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
#pragma unroll
    for (int p = 0; p < NST - 1; p++) {
        if (p < nk) {
#pragma unroll
            for (int c = 0; c < NCH; c++) issue_chunk(p, c);
        }
        cp_commit();
    }
    cp_wait<NST - 2>();
    __syncthreads();
    load_frags(0, 0, 0);
    for (int it = 0; it < nk; it++) {
        load_frags(1, it, 16);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) mma_bf16(acc[i][j], af[0][i], bf[0][j][0], bf[0][j][1]);
        cp_wait<NST - 3>();                    // stage it+1 landed (this thread's copies) ...
        __syncthreads();                       // ... for every thread, and nobody still reads the stage refilled below
        const int nxt = it + NST - 1;          // slab to fetch into the stage that was read two k steps ago
        if (nxt < nk) {
#pragma unroll
            for (int c = 0; c < NCH; c++) issue_chunk(nxt, c);
        }
        if (it + 1 < nk) load_frags(0, it + 1, 0);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < NJ; j++) mma_bf16(acc[i][j], af[1][i], bf[1][j][0], bf[1][j][1]);
        cp_commit();
    }
    cp_wait<0>();
    // bf16 output (and a bf16 saved pre-activation): the tile is staged in shared memory and leaves as full 16-byte pieces, one 512-byte row
    // segment per 32 lanes. Storing straight from the accumulators writes 16 bytes of every 32-byte sector (4 lanes x one bf16 pair), which the
    // memory system turns into read-modify-write; on the 4096 x 50304 logits that alone cost 4x in TFLOP/s.
    if (ob) {
        constexpr int STG = BN + 8;   // staged row in bf16 elements; 528 bytes: the 32-bit staging writes of a warp hit 32 different banks
        __syncthreads();              // every warp is done with the last pipeline stage: the staging area reuses it
        uint16_t* stg = reinterpret_cast<uint16_t*>(kvm_gpt_smem);
        const bool rpair = res && (ldr & 1) == 0 && ((uintptr_t)res & 3) == 0, opair = (ldo & 1) == 0 && ((uintptr_t)out & 3) == 0;
        auto pass = [&](bool want_pre) {   // want_pre: stage the pre-activation (acc*alpha + bias); else the final value
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < NJ; j++)
#pragma unroll
                    for (int hf = 0; hf < 2; hf++) {
                        const int r = wm * 64 + i * 16 + g + hf * 8, c = wn * 64 + j * 8 + 2 * t, gm = m0 + r, gn = n0 + c;
                        float v0 = 0.f, v1 = 0.f;
                        if (gm < M && gn < N) {
                            const bool two = gn + 1 < N;
                            const size_t oi = (size_t)gm * ldo + gn, ri = (size_t)gm * ldr + gn, pi = (size_t)gm * ldp + gn;
                            v0 = acc[i][j][2 * hf] * alpha; v1 = acc[i][j][2 * hf + 1] * alpha;
                            if (bias) { v0 += bias[gn]; if (two) v1 += bias[gn + 1]; }
                            if (!want_pre) {
                                if (pre && !pb) { st_s(pre, pi, v0, false); if (two) st_s(pre, pi + 1, v1, false); }
                                if (gelu) { v0 = gelu_f(v0); v1 = gelu_f(v1); }
                                if (res) {
                                    if (two && rpair) {
                                        if (rb) { const unsigned r2 = *reinterpret_cast<const unsigned*>((const uint16_t*)res + ri); v0 += __uint_as_float(r2 << 16); v1 += __uint_as_float(r2 & 0xffff0000u); }
                                        else { const float2 r2 = *reinterpret_cast<const float2*>((const float*)res + ri); v0 += r2.x; v1 += r2.y; }
                                    } else { v0 += ld_s(res, ri, rb); if (two) v1 += ld_s(res, ri + 1, rb); }
                                }
                                if (accum) {
                                    if (two && opair) { const unsigned o2 = *reinterpret_cast<const unsigned*>((const uint16_t*)out + oi); v0 += __uint_as_float(o2 << 16); v1 += __uint_as_float(o2 & 0xffff0000u); }
                                    else { v0 += ld_s(out, oi, true); if (two) v1 += ld_s(out, oi + 1, true); }
                                }
                            }
                        }
                        *reinterpret_cast<unsigned*>(stg + r * STG + c) = pack_bf16(v0, v1);
                    }
        };
        auto copy_out = [&](void* dst, int ldd) {   // staged tile -> global, 16 bytes per lane along a row; ragged edges element by element
            __syncthreads();
            for (int q = threadIdx.x; q < BM * (BN / 8); q += KVM_GPT_THREADS) {
                const int row = q / (BN / 8), c8 = q % (BN / 8), gm = m0 + row, gn = n0 + c8 * 8;
                if (gm >= M || gn >= N) continue;
                const uint16_t* src = stg + row * STG + c8 * 8;
                uint16_t* d = (uint16_t*)dst + (size_t)gm * ldd + gn;
                if (gn + 8 <= N && ((uintptr_t)d & 15) == 0) *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(src);
                else for (int e = 0; e < 8 && gn + e < N; e++) d[e] = src[e];
            }
            __syncthreads();
        };
        if (pre && pb) { pass(true); copy_out(pre, ldp); }
        pass(false);
        copy_out(out, ldo);
        return;
    }
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
