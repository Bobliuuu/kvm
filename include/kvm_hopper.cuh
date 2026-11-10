#pragma once
// Hopper (sm_90a) GEMM tiles for the KVM op library: TMA (cp.async.bulk.tensor) loads into swizzled shared memory, mbarrier
// completion, wgmma.mma_async on the tensor cores. Compiled into the op packs only for sm_90a (__CUDA_ARCH_FEAT_SM90_ALL); every other
// target (sm_80, sm_86, sm_120, plain sm_90 without the "a" features) keeps the mma.sync paths unchanged, and so does sm_90a whenever a
// precondition below fails (the caller then runs its own path).
//
//   kvmhop::mm_f32<TM, TN>   matmul_tile 1xTF32 (flags 0..15 with 32 = 128x128, 64 = 256x128; not 16 = 3xTF32 "precise"; the 64x64 tile only for K-major A and B):
//                            C[m0:m0+TM, n0:n0+TN] = alpha * op(A) op(B) with matmul_tile's epilogue (bias, pre, gelu, res, accumulate).
//                            tf32 wgmma reads only K-major operands, so an MN-major operand (A with flag 1, B without flag 2) lands
//                            through TMA in a staging area and 256 threads transpose it into the K-major layout before the wgmma.
//   kvmhop::mm_bf16          matmul_mma_bf16 (128 x 256 tile, raw bf16 operands, fp32 or bf16 out/res/pre): wgmma m64n256k16 with the
//                            transpose bits, so all four layouts run without a transpose pass.
//
// Numerics: fp32 accumulation in a fixed order (k in increasing order, one accumulator per output element), so results are bitwise
// reproducible run to run on one GPU. tf32 wgmma uses the top 19 bits of each fp32 input (truncation; the sm_80 path rounds with
// cvt.rna), which is inside matmul_tile's documented 1xTF32 tolerance (4e-3 relative to max|ref|) but NOT bitwise equal to the A100 path.
//
// Tensor maps without host work per task: kvm_hopper_host_init() (called once by the executor / test harness) encodes one template map per
// (element type, swizzle) with cuTensorMapEncodeTiled and copies them to the device. A task copies its template into a per-block slot in
// global memory and patches address, extents, row stride and box with tensormap.replace, then fences the tensormap proxy. If the host init
// did not run (kvm_tma_ready == 0) the Hopper path is skipped.
//
// Shared memory: the ops read %dynamic_smem_size and pick the pipeline depth that fits (the executor raises the launch to
// KVM_HOP_SMEM_BYTES on sm_90 through kvm_hopper_dyn_smem); a launch too small for even the minimum depth falls back.
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cudaTypedefs.h>   // CUtensorMap, PFN_cuTensorMapEncodeTiled

#define KVM_HOP_DEVI __device__ __forceinline__
#ifndef KVM_HOP_SMEM_BYTES
#define KVM_HOP_SMEM_BYTES (200 * 1024)   // dynamic shared memory the executor launches with on sm_90 (H100 opt-in limit 227 KiB)
#endif
#ifndef KVM_TMA_MAXB
#define KVM_TMA_MAXB 1024                 // blocks with a tensor-map slot (executor grid is SMs x blocks/SM: 132 on H100 SXM; the op tests
#endif                                    // launch one block per tile and raise it)

enum { KVM_TMA_F32_SW128 = 0, KVM_TMA_BF16_SW64 = 1, KVM_TMA_BF16_SW128 = 2, KVM_TMA_NKIND = 3 };
__device__ CUtensorMap kvm_tma_tmpl[KVM_TMA_NKIND];
__device__ CUtensorMap kvm_tma_slot[KVM_TMA_MAXB][2];
__device__ int kvm_tma_ready;
struct KvmTmaKey { unsigned long long base, stride; unsigned inner, outer, bi, bo, kind, valid; };
__device__ KvmTmaKey kvm_tma_key[KVM_TMA_MAXB][2];   // what each slot currently holds (valid 0 = nothing)
__device__ unsigned long long kvm_hop_count[4];   // tiles taken by the Hopper path: [0] tf32, [1] bf16, [2] tf32 fallbacks, [3] bf16 fallbacks

// ---- host side -------------------------------------------------------------------------------------------------------------------------
// Returns the dynamic shared memory the executor should launch with: `base` unless the device is sm_90+ and the Hopper path is enabled.
static inline size_t kvm_hopper_dyn_smem(size_t base) {
    int dev = 0, maj = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return base;
    if (maj != 9 || getenv("KVM_NO_HOPPER")) return base;
    return base > (size_t)KVM_HOP_SMEM_BYTES ? base : (size_t)KVM_HOP_SMEM_BYTES;
}
// Encodes the template maps and arms the device path. Returns true when the Hopper path is armed. KVM_NO_HOPPER=1 disables it (A/B runs).
static inline bool kvm_hopper_host_init() {
    int dev = 0, maj = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return false;
    int off = 0;
    if (maj != 9 || getenv("KVM_NO_HOPPER")) { cudaMemcpyToSymbol(kvm_tma_ready, &off, 4); return false; }
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
#if CUDART_VERSION >= 12050
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess || !fn) return false;
#else
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &fn, cudaEnableDefault, &q) != cudaSuccess || !fn) return false;
#endif
    auto enc = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
    void* dummy = nullptr;   // any 16-byte aligned device address: the tasks replace it
    if (cudaGetSymbolAddress(&dummy, kvm_tma_slot) != cudaSuccess) return false;
    CUtensorMap m[KVM_TMA_NKIND];
    const cuuint64_t dims[2] = {64, 64};
    const cuuint32_t estr[2] = {1, 1};
    struct K { CUtensorMapDataType t; int es; CUtensorMapSwizzle sw; cuuint32_t bi, bo; } kinds[KVM_TMA_NKIND] = {
        {CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, CU_TENSOR_MAP_SWIZZLE_128B, 32, 64},
        {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, CU_TENSOR_MAP_SWIZZLE_64B, 32, 64},
        {CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, CU_TENSOR_MAP_SWIZZLE_128B, 64, 64}};
    for (int i = 0; i < KVM_TMA_NKIND; i++) {
        const cuuint64_t str[1] = {(cuuint64_t)(64 * kinds[i].es)};
        const cuuint32_t box[2] = {kinds[i].bi, kinds[i].bo};
        if (enc(&m[i], kinds[i].t, 2, dummy, dims, str, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, kinds[i].sw,
                CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) return false;
    }
    if (cudaMemcpyToSymbol(kvm_tma_tmpl, m, sizeof(m)) != cudaSuccess) return false;
    static KvmTmaKey zero[KVM_TMA_MAXB][2];   // templates (re)written: no slot may be reused
    if (cudaMemcpyToSymbol(kvm_tma_key, zero, sizeof(zero)) != cudaSuccess) return false;
    int on = 1;
    return cudaMemcpyToSymbol(kvm_tma_ready, &on, 4) == cudaSuccess;
}
// Hopper-path tile counters since the last reset (host): tf32 taken, bf16 taken, tf32 declined, bf16 declined.
static inline void kvm_hopper_counts(unsigned long long out[4], bool reset = false) {
    cudaMemcpyFromSymbol(out, kvm_hop_count, sizeof(unsigned long long) * 4);
    if (reset) { unsigned long long z[4] = {0, 0, 0, 0}; cudaMemcpyToSymbol(kvm_hop_count, z, sizeof(z)); }
}

namespace kvmhop {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define KVM_HOPPER_DEVICE 1

#include "kvm_wgmma.inc"

KVM_HOP_DEVI unsigned s32(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
KVM_HOP_DEVI unsigned dyn_smem_size() { unsigned r; asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(r)); return r; }

// ---- mbarrier ---------------------------------------------------------------------------------------------------------------------------
KVM_HOP_DEVI void bar_init(unsigned b, unsigned n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory"); }
KVM_HOP_DEVI void bar_inval(unsigned b) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(b) : "memory"); }
KVM_HOP_DEVI void bar_expect(unsigned b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
KVM_HOP_DEVI void bar_wait(unsigned b, unsigned parity) {
    unsigned done = 0;
    while (!done)
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(done) : "r"(b), "r"(parity) : "memory");
}

// ---- tensor maps ------------------------------------------------------------------------------------------------------------------------
// One thread: slot = template `kind` with base, extents {inner, outer} (elements), row stride (bytes) and box {bi, bo} replaced. The slot
// keeps its last key: a task of the same GEMM operand (the common case: consecutive tiles of one matmul share A or B) reuses the map
// without touching it again.
KVM_HOP_DEVI void tma_make(int which, int kind, const void* base, unsigned inner, unsigned outer, unsigned long long stride, unsigned bi,
                           unsigned bo) {
    CUtensorMap* slot = &kvm_tma_slot[blockIdx.x][which];
    KvmTmaKey& key = kvm_tma_key[blockIdx.x][which];
    const KvmTmaKey want{(unsigned long long)(uintptr_t)base, stride, inner, outer, bi, bo, (unsigned)kind, 1u};
    if (key.base == want.base && key.stride == want.stride && key.inner == want.inner && key.outer == want.outer && key.bi == want.bi &&
        key.bo == want.bo && key.kind == want.kind && key.valid == 1u)
        return;
    const int4* s = reinterpret_cast<const int4*>(&kvm_tma_tmpl[kind]);
    int4* d = reinterpret_cast<int4*>(slot);
#pragma unroll
    for (int i = 0; i < 8; i++) d[i] = s[i];
    asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;" ::"l"(slot), "l"(base) : "memory");
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 0, %1;" ::"l"(slot), "r"(inner) : "memory");
    asm volatile("tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(slot), "r"(outer) : "memory");
    asm volatile("tensormap.replace.tile.global_stride.global.b1024.b64 [%0], 0, %1;" ::"l"(slot), "l"(stride) : "memory");
    asm volatile("tensormap.replace.tile.box_dim.global.b1024.b32 [%0], 0, %1;" ::"l"(slot), "r"(bi) : "memory");
    asm volatile("tensormap.replace.tile.box_dim.global.b1024.b32 [%0], 1, %1;" ::"l"(slot), "r"(bo) : "memory");
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" ::: "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(slot) : "memory");
    key = want;
}
// 2-D TMA load of the box at element coordinates (c0 inner, c1 outer) into shared address dst, completing on mbarrier bar.
KVM_HOP_DEVI void tma_load(unsigned dst, const CUtensorMap* map, int c0, int c1, unsigned bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}

// ---- wgmma ------------------------------------------------------------------------------------------------------------------------------
KVM_HOP_DEVI void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
KVM_HOP_DEVI void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> KVM_HOP_DEVI void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
template <int N> KVM_HOP_DEVI void wg_hold(float (&d)[N]) {   // operand fence: keep the accumulators in place around the async section
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i])::"memory");
}
// Shared-memory matrix descriptor. swz: 1 = 128B, 2 = 64B, 3 = 32B. lbo / sbo in bytes.
KVM_HOP_DEVI uint64_t desc(unsigned addr, unsigned lbo, unsigned sbo, int swz) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)((lbo >> 4) & 0x3FFF) << 16) | ((uint64_t)((sbo >> 4) & 0x3FFF) << 32) |
           ((uint64_t)swz << 62);
}

// Transpose an MN-major fp32 operand tile, staged by TMA as R/32 boxes of [32 k][32 mn] (128-byte swizzle, box j at j*4096), into the
// K-major 128-byte-swizzled layout wgmma reads ([R mn][32 k]). All 256 threads; one float4 (4 k of one mn) written per item.
template <int R>
KVM_HOP_DEVI void transpose_f32(unsigned src, unsigned dst) {
    for (int q = threadIdx.x; q < R * 8; q += 256) {
        const int mn = q % R, kc = q / R, j = mn >> 5, c = mn & 31;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int k = kc * 4 + e;
            const unsigned a = src + j * 4096 + k * 128 + ((((c >> 2) ^ (k & 7))) << 4) + (c & 3) * 4;
            asm volatile("ld.shared.f32 %0, [%1];" : "=f"(v[e]) : "r"(a));
        }
        const unsigned o = dst + mn * 128 + (((kc ^ (mn & 7))) << 4);
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(o), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]) : "memory");
    }
}

KVM_HOP_DEVI float gelu(float x) {   // same formula as kvmgpt::gelu_f (tanh approximation, full-precision tanhf)
    const float c = 0.7978845608028654f;
    return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// Epilogue of one accumulator pair at (gm, gn), (gm, gn+1): matmul_tile order (alpha, bias, pre, gelu, res, accumulate), fp32 storage.
KVM_HOP_DEVI void epi_f32(float v0, float v1, int gm, int gn, int M, int N, float alpha, const float* bias, float* pre, int ldp, bool gelu_on,
                          const float* res, int ldr, float* out, int ldo, bool accum, bool pair) {
    if (gm >= M || gn >= N) return;
    float v[2] = {v0 * alpha, v1 * alpha};
    const int nv = gn + 1 < N ? 2 : 1;
    if (pair && nv == 2) {
        if (bias) { const float2 b = *reinterpret_cast<const float2*>(bias + gn); v[0] += b.x; v[1] += b.y; }
        if (pre) *reinterpret_cast<float2*>(pre + (size_t)gm * ldp + gn) = make_float2(v[0], v[1]);
        if (gelu_on) { v[0] = gelu(v[0]); v[1] = gelu(v[1]); }
        if (res) { const float2 r = *reinterpret_cast<const float2*>(res + (size_t)gm * ldr + gn); v[0] += r.x; v[1] += r.y; }
        float2* o = reinterpret_cast<float2*>(out + (size_t)gm * ldo + gn);
        if (accum) { const float2 p = *o; v[0] += p.x; v[1] += p.y; }
        *o = make_float2(v[0], v[1]);
        return;
    }
    for (int e = 0; e < nv; e++) {
        float x = v[e];
        if (bias) x += bias[gn + e];
        if (pre) pre[(size_t)gm * ldp + gn + e] = x;
        if (gelu_on) x = gelu(x);
        if (res) x += res[(size_t)gm * ldr + gn + e];
        float* o = out + (size_t)gm * ldo + gn + e;
        *o = accum ? *o + x : x;
    }
}

KVM_HOP_DEVI bool tma_ok(const void* p, int ld, int es) { return ((uintptr_t)p & 15) == 0 && ((size_t)ld * es) % 16 == 0 && ld > 0; }
// Usable dynamic shared memory: the launch size minus a worst-case 1024-byte alignment slack and the 1024-byte barrier block, so the
// pipeline depth (and a decline) depends only on the launch size, never on where the dynamic window happens to start.
KVM_HOP_DEVI unsigned smem_avail() { const unsigned s = dyn_smem_size(); return s > 2048 ? s - 2048 : 0; }

// ---- tf32 tile ---------------------------------------------------------------------------------------------------------------------------
// Warpgroup split: TM == 64: both warpgroups take rows 0..63, warpgroup w columns [32w, 32w+32) (m64n32k8). TM == 128: warpgroup w rows
// [64w, 64w+64), all TN = 128 columns (m64n128k8). TM == 256: warpgroup w rows [128w, 128w+128) as two m64n128k8.
// Pipeline: NST TMA stages (prefetch NST-1 slabs ahead); the wgmma group of slab it overlaps the transpose and the TMA of later slabs
// (wait_group 1); a stage is refilled once the group that read it has completed. MN-major operands are transposed into a double buffer.
template <int TM, int TN>
KVM_HOP_DEVI bool mm_f32(const float* A, int lda, bool ta, const float* B, int ldb, bool tb, float* out, int ldo, const float* bias,
                         const float* res, int ldr, float* pre, int ldp, int m0, int n0, int M, int N, int K, float alpha, bool gelu_on, bool accum,
                         unsigned char* smem) {
    static_assert((TM == 64 && TN == 64) || (TM == 128 && TN == 128) || (TM == 256 && TN == 128), "tile");
    constexpr int BK = 32, ABY = TM * 128, BBY = TN * 128, STG = ABY + BBY;
    constexpr int WN = TM == 64 ? 32 : 128, MI = TM == 256 ? 2 : 1, NACC = WN / 2;
    if (!kvm_tma_ready || blockIdx.x >= KVM_TMA_MAXB || !tma_ok(A, lda, 4) || !tma_ok(B, ldb, 4)) return false;
    // 64x64: only the all-K-major layout (flag 2) gains on an H100 (73 vs 60 TFLOP/s at 4096^3; NN ties, TN loses 38 vs 48 to the
    // shared-memory transpose: results/2026-11-09-h100-hopper); the larger tiles win in every layout measured.
    if (TM == 64 && (ta || !tb)) return false;
    const unsigned tbytes = 2 * ((ta ? ABY : 0) + (!tb ? BBY : 0));   // double-buffered transposed copies of MN-major operands
    const unsigned avail = smem_avail();
    int nst = 0;
    while (nst < 6 && (unsigned)(nst + 1) * STG + tbytes <= avail) nst++;
    if (nst < 2) return false;
    const unsigned base = s32(smem), sb = (base + 1023) & ~1023u;
    const unsigned bars = sb, st0 = sb + 1024, tA = st0 + nst * STG, tB = tA + (ta ? 2 * ABY : 0);
    const int tid = threadIdx.x, wg = tid >> 7, nk = (K + BK - 1) / BK;
    const CUtensorMap* mA = &kvm_tma_slot[blockIdx.x][0];
    const CUtensorMap* mB = &kvm_tma_slot[blockIdx.x][1];
    if (tid == 0) {
        // A: K-major [M][K] (box 32 k x TM rows), or MN-major [K][M] (box 32 m x 32 k, TM/32 boxes)
        if (!ta) tma_make(0, KVM_TMA_F32_SW128, A, K, M, (unsigned long long)lda * 4, 32, TM);
        else tma_make(0, KVM_TMA_F32_SW128, A, M, K, (unsigned long long)lda * 4, 32, 32);
        if (tb) tma_make(1, KVM_TMA_F32_SW128, B, K, N, (unsigned long long)ldb * 4, 32, TN);
        else tma_make(1, KVM_TMA_F32_SW128, B, N, K, (unsigned long long)ldb * 4, 32, 32);
        for (int s = 0; s < nst; s++) bar_init(bars + 8 * s, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int p = 0; p < nst && p < nk; p++) {
            const unsigned b = bars + 8 * p, da = st0 + p * STG, db = da + ABY;
            bar_expect(b, STG);
            if (!ta) tma_load(da, mA, p * BK, m0, b);
            else for (int j = 0; j < TM / 32; j++) tma_load(da + j * 4096, mA, m0 + 32 * j, p * BK, b);
            if (tb) tma_load(db, mB, p * BK, n0, b);
            else for (int j = 0; j < TN / 32; j++) tma_load(db + j * 4096, mB, n0 + 32 * j, p * BK, b);
        }
    }
    __syncthreads();
    float acc[MI][NACC];
#pragma unroll
    for (int i = 0; i < MI; i++)
#pragma unroll
        for (int e = 0; e < NACC; e++) acc[i][e] = 0.f;
    for (int it = 0; it < nk; it++) {
        const int s = it % nst;
        bar_wait(bars + 8 * s, (it / nst) & 1);
        unsigned aS = st0 + s * STG, bS = aS + ABY;
        if (ta || !tb) {   // the copy written now was last read by group it-2, complete since wait_group 1 of iteration it-1
            const unsigned tAi = tA + (it & 1) * ABY, tBi = tB + (it & 1) * BBY;
            if (ta) transpose_f32<TM>(aS, tAi);
            if (!tb) transpose_f32<TN>(bS, tBi);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncthreads();
            if (ta) aS = tAi;
            if (!tb) bS = tBi;
        }
#pragma unroll
        for (int i = 0; i < MI; i++) wg_hold(acc[i]);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 8; kk++) {
#pragma unroll
            for (int i = 0; i < MI; i++) {
                const int r0 = TM == 64 ? 0 : (TM == 128 ? 64 * wg : 128 * wg + 64 * i);
                const int c0 = TM == 64 ? 32 * wg : 0;
                const uint64_t dA = desc(aS + r0 * 128 + kk * 32, 16, 1024, 1), dB = desc(bS + c0 * 128 + kk * 32, 16, 1024, 1);
                if constexpr (WN == 32) wgmma_m64n32k8_tf32(acc[i], dA, dB);
                else wgmma_m64n128k8_tf32(acc[i], dA, dB);
            }
        }
        wg_commit();
#pragma unroll
        for (int i = 0; i < MI; i++) wg_hold(acc[i]);
        wg_wait<1>();      // group it-1 is complete in this warpgroup ...
        __syncthreads();   // ... and in the other one: stage (it-1) % nst is free
        if (tid == 0 && it >= 1 && it - 1 + nst < nk) {
            const int p = it - 1 + nst, ps = (it - 1) % nst;
            const unsigned b = bars + 8 * ps, da = st0 + ps * STG, db = da + ABY;
            bar_expect(b, STG);
            if (!ta) tma_load(da, mA, p * BK, m0, b);
            else for (int j = 0; j < TM / 32; j++) tma_load(da + j * 4096, mA, m0 + 32 * j, p * BK, b);
            if (tb) tma_load(db, mB, p * BK, n0, b);
            else for (int j = 0; j < TN / 32; j++) tma_load(db + j * 4096, mB, n0 + 32 * j, p * BK, b);
        }
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < MI; i++) wg_hold(acc[i]);
    __syncthreads();
    if (tid == 0)
        for (int s = 0; s < nst; s++) bar_inval(bars + 8 * s);
    // epilogue from the wgmma D fragments
    const int wi = (tid & 127) >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const bool pair = (((uintptr_t)out | (uintptr_t)res | (uintptr_t)pre | (uintptr_t)bias) & 7) == 0 && (ldo & 1) == 0 && (ldr & 1) == 0 && (ldp & 1) == 0;
#pragma unroll
    for (int i = 0; i < MI; i++) {
        const int r0 = TM == 64 ? 0 : (TM == 128 ? 64 * wg : 128 * wg + 64 * i), c0 = TM == 64 ? 32 * wg : 0;
#pragma unroll
        for (int j = 0; j < WN / 8; j++)
#pragma unroll
            for (int hf = 0; hf < 2; hf++)
                epi_f32(acc[i][4 * j + 2 * hf], acc[i][4 * j + 2 * hf + 1], m0 + r0 + 16 * wi + g + 8 * hf, n0 + c0 + 8 * j + 2 * t, M, N, alpha, bias,
                        pre, ldp, gelu_on, res, ldr, out, ldo, accum, pair);
    }
    __syncthreads();   // the next op may reuse shared memory
    return true;
}

// ---- bf16 tile (matmul_mma_bf16: 128 x 256) ----------------------------------------------------------------------------------------------
KVM_HOP_DEVI float lds_(const void* p, size_t i, bool bf) { return bf ? __uint_as_float((uint32_t)((const uint16_t*)p)[i] << 16) : ((const float*)p)[i]; }
KVM_HOP_DEVI uint16_t bf16r(float x) {   // round to nearest even (finite inputs; NaN keeps a quiet NaN)
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}
KVM_HOP_DEVI void sts_(void* p, size_t i, float x, bool bf) { if (bf) ((uint16_t*)p)[i] = bf16r(x); else ((float*)p)[i] = x; }

// Warpgroup w owns rows [64w, 64w+64) x all 256 columns (one m64n256k16 per k16 step, 128 accumulators per thread). BK 64 (128-byte rows) when
// two 48 KiB stages fit, else BK 32 (24 KiB stages; the 96 KiB launch of the sm_80 build).
//   K-major  [rows][K]: one box BK k x rows; BK 64: 128-byte swizzle (SBO 1024), BK 32: 64-byte swizzle (SBO 512); k16 step = +32 bytes.
//   MN-major [K][rows]: boxes 64 mn x BK k (128-byte swizzle) at j * (BK*128) bytes; LBO = BK*128 (next 64-wide mn chunk), SBO 1024 (next
//                       8 k rows); k16 step = +2048 bytes. Warpgroup w's A rows are box w.
template <bool TA, bool TB, int BK>
KVM_HOP_DEVI void mm_bf16_body(const uint16_t* A, int lda, const uint16_t* B, int ldb, int m0, int n0, int M, int N, int K, int nst, unsigned sb,
                               float (&acc)[128]) {
    constexpr int TM = 128, TN = 256, ABY = TM * BK * 2, BBY = TN * BK * 2, STG = ABY + BBY, CH = BK * 128;
    constexpr int KSW = BK == 64 ? 1 : 2, KSBO = BK == 64 ? 1024 : 512, KROW = BK * 2;
    const int kmaj = BK == 64 ? KVM_TMA_BF16_SW128 : KVM_TMA_BF16_SW64;
    const unsigned bars = sb, st0 = sb + 1024;
    const int tid = threadIdx.x, wg = tid >> 7, nk = (K + BK - 1) / BK;
    const CUtensorMap* mA = &kvm_tma_slot[blockIdx.x][0];
    const CUtensorMap* mB = &kvm_tma_slot[blockIdx.x][1];
    auto issue = [&](int p, int ps) {
        const unsigned b = bars + 8 * ps, da = st0 + ps * STG, db = da + ABY;
        bar_expect(b, STG);
        if (!TA) tma_load(da, mA, p * BK, m0, b);
        else for (int j = 0; j < TM / 64; j++) tma_load(da + j * CH, mA, m0 + 64 * j, p * BK, b);
        if (TB) tma_load(db, mB, p * BK, n0, b);
        else for (int j = 0; j < TN / 64; j++) tma_load(db + j * CH, mB, n0 + 64 * j, p * BK, b);
    };
    if (tid == 0) {
        if (!TA) tma_make(0, kmaj, A, K, M, (unsigned long long)lda * 2, BK, TM);
        else tma_make(0, KVM_TMA_BF16_SW128, A, M, K, (unsigned long long)lda * 2, 64, BK);
        if (TB) tma_make(1, kmaj, B, K, N, (unsigned long long)ldb * 2, BK, TN);
        else tma_make(1, KVM_TMA_BF16_SW128, B, N, K, (unsigned long long)ldb * 2, 64, BK);
        for (int s = 0; s < nst; s++) bar_init(bars + 8 * s, 1);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        for (int p = 0; p < nst && p < nk; p++) issue(p, p);
    }
    __syncthreads();
    for (int it = 0; it < nk; it++) {
        const int s = it % nst;
        bar_wait(bars + 8 * s, (it / nst) & 1);
        const unsigned aS = st0 + s * STG, bS = aS + ABY;
        wg_hold(acc);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            const uint64_t dA = !TA ? desc(aS + 64 * wg * KROW + kk * 32, 16, KSBO, KSW) : desc(aS + wg * CH + kk * 2048, CH, 1024, 1);
            const uint64_t dB = TB ? desc(bS + kk * 32, 16, KSBO, KSW) : desc(bS + kk * 2048, CH, 1024, 1);
            wgmma_m64n256k16_bf16<TA ? 1 : 0, TB ? 0 : 1>(acc, dA, dB);
        }
        wg_commit();
        wg_hold(acc);
        wg_wait<1>();
        __syncthreads();
        if (tid == 0 && it >= 1 && it - 1 + nst < nk) issue(it - 1 + nst, (it - 1) % nst);
    }
    wg_wait<0>();
    wg_hold(acc);
    __syncthreads();
    if (tid == 0)
        for (int s = 0; s < nst; s++) bar_inval(bars + 8 * s);
}

template <bool TA, bool TB>
KVM_HOP_DEVI bool mm_bf16(const uint16_t* A, int lda, const uint16_t* B, int ldb, void* out, int ldo, bool ob, const float* bias,
                          const void* res, int ldr, bool rb, void* pre, int ldp, bool pb, int m0, int n0, int M, int N, int K, float alpha,
                          bool gelu_on, bool accum, unsigned char* smem) {
    constexpr int TM = 128, TN = 256;
    if (!kvm_tma_ready || blockIdx.x >= KVM_TMA_MAXB || !tma_ok(A, lda, 2) || !tma_ok(B, ldb, 2)) return false;
    const unsigned avail = smem_avail(), sb = (s32(smem) + 1023) & ~1023u;
    constexpr unsigned S64 = (TM + TN) * 64 * 2, S32 = (TM + TN) * 32 * 2;
    float acc[128];
#pragma unroll
    for (int e = 0; e < 128; e++) acc[e] = 0.f;
    if (2 * S64 <= avail) {
        int nst = 0;
        while (nst < 6 && (nst + 1) * S64 <= avail) nst++;
        mm_bf16_body<TA, TB, 64>(A, lda, B, ldb, m0, n0, M, N, K, nst, sb, acc);
    } else if (2 * S32 <= avail) {
        int nst = 0;
        while (nst < 8 && (nst + 1) * S32 <= avail) nst++;
        mm_bf16_body<TA, TB, 32>(A, lda, B, ldb, m0, n0, M, N, K, nst, sb, acc);
    } else {
        return false;
    }
    const int tid = threadIdx.x, wg = tid >> 7, wi = (tid & 127) >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const bool pair = (((uintptr_t)out | (uintptr_t)res | (uintptr_t)pre | (uintptr_t)bias) & 7) == 0 && (ldo & 1) == 0 && (ldr & 1) == 0 && (ldp & 1) == 0;
#pragma unroll
    for (int j = 0; j < TN / 8; j++)
#pragma unroll
        for (int hf = 0; hf < 2; hf++) {
            const int gm = m0 + 64 * wg + 16 * wi + g + 8 * hf, gn = n0 + 8 * j + 2 * t;
            if (gm >= M || gn >= N) continue;
            const int nv = gn + 1 < N ? 2 : 1;
            if (pair && nv == 2) {   // column pair as one 64-bit (fp32) / 32-bit (bf16x2) access per operand
                float v0 = acc[4 * j + 2 * hf] * alpha, v1 = acc[4 * j + 2 * hf + 1] * alpha;
                if (bias) { const float2 b2 = *reinterpret_cast<const float2*>(bias + gn); v0 += b2.x; v1 += b2.y; }
                const size_t oi = (size_t)gm * ldo + gn, ri = (size_t)gm * ldr + gn, pi = (size_t)gm * ldp + gn;
                if (pre) {
                    if (pb) *reinterpret_cast<unsigned*>((uint16_t*)pre + pi) = (unsigned)bf16r(v0) | ((unsigned)bf16r(v1) << 16);
                    else *reinterpret_cast<float2*>((float*)pre + pi) = make_float2(v0, v1);
                }
                if (gelu_on) { v0 = gelu(v0); v1 = gelu(v1); }
                if (res) {
                    if (rb) { const unsigned r2 = *reinterpret_cast<const unsigned*>((const uint16_t*)res + ri); v0 += __uint_as_float(r2 << 16); v1 += __uint_as_float(r2 & 0xffff0000u); }
                    else { const float2 r2 = *reinterpret_cast<const float2*>((const float*)res + ri); v0 += r2.x; v1 += r2.y; }
                }
                if (ob) {
                    unsigned* o = reinterpret_cast<unsigned*>((uint16_t*)out + oi);
                    if (accum) { const unsigned o2 = *o; v0 += __uint_as_float(o2 << 16); v1 += __uint_as_float(o2 & 0xffff0000u); }
                    *o = (unsigned)bf16r(v0) | ((unsigned)bf16r(v1) << 16);
                } else {
                    float2* o = reinterpret_cast<float2*>((float*)out + oi);
                    if (accum) { const float2 o2 = *o; v0 += o2.x; v1 += o2.y; }
                    *o = make_float2(v0, v1);
                }
                continue;
            }
            for (int e = 0; e < nv; e++) {
                float x = acc[4 * j + 2 * hf + e] * alpha;
                if (bias) x += bias[gn + e];
                if (pre) sts_(pre, (size_t)gm * ldp + gn + e, x, pb);
                if (gelu_on) x = gelu(x);
                if (res) x += lds_(res, (size_t)gm * ldr + gn + e, rb);
                const size_t oi = (size_t)gm * ldo + gn + e;
                if (accum) x += lds_(out, oi, ob);
                sts_(out, oi, x, ob);
            }
        }
    __syncthreads();
    return true;
}

#endif  // sm_90a

}  // namespace kvmhop
