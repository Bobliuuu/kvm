// matmul_mma_bf16 (bf16 tensor-core tile) against cuBLAS bf16 (cublasGemmEx, fp32 accumulate) in ONE process, plus a correctness check
// against cuBLAS on the same bf16 operands for every configuration. Each CTA runs one 128 x 256 tile exactly as the executor would
// (same device function, same dynamic shared memory, 256 threads); the grid is one CTA per tile, so wave quantisation is included.
//   build: nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -lineinfo -I../../include -I. bench_mma_bf16.cu -lcublas
//   env:   BENCH_REPS=n (median of n timings, default 7), MMA_SET=big|gpt|all (default all), MMA_OUT=f32|bf16 (default bf16)
// A100 peaks: bf16 tensor 312 TFLOP/s, TF32 156.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include "kvm_ctx_shim.cuh"
#include "kvm_ops_gpt.cuh"
#include "kvm_ops_extra.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

#ifndef MMA_MINB
#define MMA_MINB 1
#endif
enum Which { MMA = 0, OLD_BF16 = 1 };
__global__ void __launch_bounds__(256, MMA_MINB) k_mma(KvmCtx c, const int32_t* w, int words) { kvmextra::matmul_mma_bf16(c, w + (size_t)blockIdx.x * words); }
__global__ void __launch_bounds__(256, 1) k_old(KvmCtx c, const int32_t* w, int words) { kvmextra::matmul_tile_bf16(c, w + (size_t)blockIdx.x * words); }

static double med(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
template <typename F> static std::vector<double> time_ms(F f, int iters) {
    static const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 7;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) f();
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); for (int i = 0; i < iters; i++) f(); cudaEventRecord(b);
        CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); CK(cudaGetLastError()); t.push_back(ms / iters);
    }
    return t;
}
template <typename F, typename G> static void time_pair(F f, G g, int iters, std::vector<double>& tf, std::vector<double>& tg) {   // alternate f and g reps so clock drift hits both alike
    static const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 7;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) { f(); g(); }
    CK(cudaDeviceSynchronize());
    for (int r = 0; r < reps; r++) {
        float ms;
        cudaEventRecord(a); for (int i = 0; i < iters; i++) f(); cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b); tf.push_back(ms / iters);
        cudaEventRecord(a); for (int i = 0; i < iters; i++) g(); cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b); tg.push_back(ms / iters);
    }
    CK(cudaGetLastError());
}
__global__ void reduce_splits(const float* part, size_t mn, int s, uint16_t* out) {   // fixed-order sum of s fp32 partial tiles -> bf16
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < mn; i += (size_t)gridDim.x * blockDim.x) {
        float v = 0.f; for (int q = 0; q < s; q++) v += part[q * mn + i];
        out[i] = kvmgpt::bf16_bits(v);
    }
}
__global__ void fill_bf16(uint16_t* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
        p[i] = kvmgpt::bf16_bits(((x & 0xffff) / 65536.f - 0.5f));
    }
}
__global__ void widen(const uint16_t* s, float* d, size_t n) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) d[i] = __uint_as_float((uint32_t)s[i] << 16); }
static cublasHandle_t g_h;
static bool g_bf_out = true;
static int g_fail = 0;

static void run(int M, int N, int K, int ta, int tb, Which which, const char* tag) {
    uint16_t *A, *B, *Cb; float *Cf, *R;
    const int pad = getenv("MMA_LDPAD") ? atoi(getenv("MMA_LDPAD")) : 0;   // diagnostic: leading dimension = extent + pad elements
    const int lda_ = (ta ? M : K) + pad, ldb_ = (tb ? K : N) + pad;
    CK(cudaMalloc(&A, (size_t)(ta ? K : M) * lda_ * 2)); CK(cudaMalloc(&B, (size_t)(tb ? N : K) * ldb_ * 2)); CK(cudaMalloc(&Cb, (size_t)M * N * 2 + 16)); CK(cudaMalloc(&Cf, (size_t)M * N * 4 + 16)); CK(cudaMalloc(&R, (size_t)M * N * 4));
    CK(cudaMemset(Cb, 0, (size_t)M * N * 2)); CK(cudaMemset(Cf, 0, (size_t)M * N * 4));
    fill_bf16<<<1024, 256>>>(A, (size_t)(ta ? K : M) * lda_, 1); fill_bf16<<<1024, 256>>>(B, (size_t)(tb ? N : K) * ldb_, 2);
    const int splits = (getenv("MMA_SPLITK") && ta && !tb && g_bf_out) ? atoi(getenv("MMA_SPLITK")) : 1;   // split-K: s partial fp32 tiles from k ranges of whole 32-slabs, fixed-order reduce kernel
    const int kslabs = (K + 31) / 32, kchunk = ((kslabs + splits - 1) / splits) * 32;
    int ns = 0; for (int q = 0; q < splits; q++) if (q * kchunk < K) ns++;
    float* part = nullptr; if (splits > 1) CK(cudaMalloc(&part, (size_t)ns * M * N * 4));
    std::vector<KvmBuf> tabv(3 + 3 * (splits > 1 ? ns : 0));
    KvmBuf* tab = tabv.data();
    tab[0].p = A; tab[0].rows = ta ? K : M; tab[0].cols = ta ? M : K; tab[0].dtype = 2;
    tab[1].p = B; tab[1].rows = tb ? N : K; tab[1].cols = tb ? K : N; tab[1].dtype = 2;
    tab[2].p = g_bf_out ? (void*)Cb : (void*)Cf; tab[2].rows = M; tab[2].cols = N; tab[2].dtype = g_bf_out ? 2 : 0;
    tab[0].ld = lda_; tab[1].ld = ldb_; tab[2].ld = tab[2].cols;
    if (splits > 1) for (int q = 0; q < ns; q++) {   // buffers 3+3q: A_q, B_q, partial_q (A and B are views at k offset q*kchunk)
        const int k0 = q * kchunk, kq = std::min(kchunk, K - k0);
        tab[3 + 3 * q] = tab[0]; tab[4 + 3 * q] = tab[1];
        tab[3 + 3 * q].p = ta ? (void*)(A + (size_t)k0 * lda_) : (void*)(A + k0); tab[3 + 3 * q].rows = ta ? kq : M; tab[3 + 3 * q].cols = ta ? M : kq;
        tab[4 + 3 * q].p = tb ? (void*)(B + k0) : (void*)(B + (size_t)k0 * ldb_); tab[4 + 3 * q].rows = tb ? N : kq; tab[4 + 3 * q].cols = tb ? kq : N;
        tab[5 + 3 * q] = tab[2]; tab[5 + 3 * q].p = part + (size_t)q * M * N; tab[5 + 3 * q].dtype = 0; tab[5 + 3 * q].ld = N;
    }
    KvmBuf* tab_d; CK(cudaMalloc(&tab_d, tabv.size() * sizeof(KvmBuf))); CK(cudaMemcpy(tab_d, tab, tabv.size() * sizeof(KvmBuf), cudaMemcpyHostToDevice));
    int32_t* err; CK(cudaMalloc(&err, 4)); CK(cudaMemset(err, 0, 4));
    const int bm = which == MMA ? KVM_MMA_BM : 64, bn = which == MMA ? KVM_MMA_BN : 64, fl = ta | (tb << 1);
    (void)0;
    const float alpha = 1.f; int32_t afw; memcpy(&afw, &alpha, 4);
    std::vector<int32_t> w; int nt = 0;
    const bool n_outer = getenv("MMA_NOUTER") != nullptr;   // tile order: m outer (default, as the frontend lowers) or n outer (consecutive tiles share a B column block)
    const int gm_grp = getenv("MMA_GROUPM") ? atoi(getenv("MMA_GROUPM")) : 0;   // grouped order: GROUPM m-blocks sweep all n before the next group (L2 reuse of both operands)
    auto emit = [&](int m0, int n0, int q = -1) { int32_t a[13] = {q >= 0 ? 3 + 3 * q : 0, q >= 0 ? 4 + 3 * q : 1, q >= 0 ? 5 + 3 * q : 2, -1, -1, -1, m0, n0, M, N, q >= 0 ? std::min(kchunk, K - q * kchunk) : K, fl, afw}; w.insert(w.end(), a, a + 13); nt++; };
    const int nsp = splits > 1 ? ns : 1;
    auto emit_all = [&](int m0, int n0) { for (int q = 0; q < nsp; q++) emit(m0, n0, splits > 1 ? q : -1); };
    if (gm_grp) { for (int g0 = 0; g0 < M; g0 += gm_grp * bm) for (int n0 = 0; n0 < N; n0 += bn) for (int m0 = g0; m0 < std::min(M, g0 + gm_grp * bm); m0 += bm) emit_all(m0, n0); }
    else if (n_outer) { for (int n0 = 0; n0 < N; n0 += bn) for (int m0 = 0; m0 < M; m0 += bm) emit_all(m0, n0); }
    else { for (int m0 = 0; m0 < M; m0 += bm) for (int n0 = 0; n0 < N; n0 += bn) emit_all(m0, n0); }
    const bool same = getenv("MMA_SAME") != nullptr;   // diagnostic: every CTA computes tile (0,0): data stays in L2, wave quantisation vanishes
    if (same) { const int total = 108 * 6; w.clear(); nt = total; for (int i = 0; i < total; i++) { int32_t a[13] = {0, 1, 2, -1, -1, -1, 0, 0, M, N, K, fl, afw}; w.insert(w.end(), a, a + 13); } }
    int32_t* wd; CK(cudaMalloc(&wd, w.size() * 4)); CK(cudaMemcpy(wd, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    KvmCtx c{tab_d, 3, err};
    const int smem = MMA_MINB > 1 ? std::max(KVM_MMA_SMEM, KVM_MMA_BM * (KVM_MMA_BN + 8) * 2) : std::max(kvm_gpt_smem_bytes(64), KVM_MMA_SMEM);
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); CK(cudaFuncSetAttribute(k_old, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); attr = true; }
    auto kvm = [&] { if (which == MMA) k_mma<<<nt, 256, smem>>>(c, wd, 13); else k_old<<<nt, 256, smem>>>(c, wd, 13); if (splits > 1) reduce_splits<<<432, 256>>>(part, (size_t)M * N, ns, Cb); };
    kvm(); CK(cudaDeviceSynchronize());
    int e; CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
    // reference: cuBLAS bf16 inputs, fp32 accumulate, fp32 out (column-major view: C^T = op(B)^T op(A)^T)
    const float one = 1.f, zero = 0.f;
    auto cb = [&] {
        cublasGemmEx(g_h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16BF, ldb_, A, CUDA_R_16BF, lda_, &zero, R,
                     CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    cb(); CK(cudaDeviceSynchronize());
    std::vector<float> hc((size_t)M * N), hr((size_t)M * N);
    if (g_bf_out) { float* tmp; CK(cudaMalloc(&tmp, (size_t)M * N * 4)); widen<<<1024, 256>>>(Cb, tmp, (size_t)M * N); CK(cudaMemcpy(hc.data(), tmp, hc.size() * 4, cudaMemcpyDeviceToHost)); cudaFree(tmp); }
    else CK(cudaMemcpy(hc.data(), Cf, hc.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(), R, hr.size() * 4, cudaMemcpyDeviceToHost));
    double me = 0, mx = 0; for (size_t i = 0; i < hc.size(); i++) { me = std::max(me, (double)fabsf(hc[i] - hr[i])); mx = std::max(mx, (double)fabsf(hr[i])); }
    const double tol = g_bf_out ? 5e-3 : 1e-5 * sqrt((double)K) + 1e-5;   // bf16 output: one rounding (2^-9 relative to the value); fp32: summation order only
    const bool ok = me / mx < tol && e == 0;
    if (!ok) g_fail++;
    const double flops = same ? 2.0 * bm * bn * K * nt : 2.0 * M * N * K;
    std::vector<double> tk, tc; time_pair(kvm, cb, 10, tk, tc);
    const double mk = med(tk), mc = med(tc);
    printf("%-9s %-5s %c%c %5dx%5dx%5d tile %3dx%-3d sk%d kvm %6.1f [%6.1f..%6.1f] cuBLAS %6.1f [%6.1f..%6.1f] TFLOP/s ratio %.3f err %.1e %s dev_err %d\n", tag, which == MMA ? "mma" : "old64",
           ta ? 'T' : 'N', tb ? 'T' : 'N', M, N, K, bm, bn, splits, flops / mk * 1e-9, flops / *std::max_element(tk.begin(), tk.end()) * 1e-9, flops / *std::min_element(tk.begin(), tk.end()) * 1e-9,
           flops / mc * 1e-9, flops / *std::max_element(tc.begin(), tc.end()) * 1e-9, flops / *std::min_element(tc.begin(), tc.end()) * 1e-9, mc / mk, me / mx, ok ? "ok" : "FAIL", e);
    fflush(stdout);
    cudaFree(A); cudaFree(B); cudaFree(Cb); cudaFree(Cf); cudaFree(R); cudaFree(tab_d); cudaFree(err); cudaFree(wd);
}
int main() {
    cublasCreate(&g_h);
    const char* set = getenv("MMA_SET") ? getenv("MMA_SET") : "all";
    g_bf_out = !(getenv("MMA_OUT") && !strcmp(getenv("MMA_OUT"), "f32"));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device %s SMs %d  out=%s\n", pr.name, pr.multiProcessorCount, g_bf_out ? "bf16" : "f32");
    if (const char* cu = getenv("MMA_SHAPE")) {   // MMA_SHAPE=M,N,K: just this shape, all four transpose combinations (MMA_ONLY_NN=1: NN only)
        int M, N, K; if (sscanf(cu, "%d,%d,%d", &M, &N, &K) == 3) {
            for (int o = 0; o < 4; o++) { if (getenv("MMA_ONLY_NN") && o) break; run(M, N, K, o & 1, o >> 1, MMA, "custom"); }
            printf("%s (%d failed)\n", g_fail ? "FAIL" : "OK", g_fail); return g_fail != 0;
        }
    }
    if (strcmp(set, "gpt")) {
        for (int o = 0; o < 4; o++) run(4096, 4096, 4096, o & 1, o >> 1, MMA, "4096^3");
        run(4096, 4096, 4096, 0, 0, OLD_BF16, "4096^3");
        run(8192, 8192, 8192, 0, 0, MMA, "8192^3");
    }
    if (strcmp(set, "big")) {
        struct S { const char* nm; int M, N, K; } fwd[] = {{"qkvo", 4096, 768, 768}, {"fc1", 4096, 3072, 768}, {"fc2", 4096, 768, 3072}, {"head", 4096, 50304, 768}};
        for (auto s : fwd) {   // forward NN, dgrad NT (dY * W^T), wgrad TN (X^T * dY)
            run(s.M, s.N, s.K, 0, 0, MMA, s.nm);
            run(s.M, s.K, s.N, 0, 1, MMA, s.nm);
            run(s.K, s.N, s.M, 1, 0, MMA, s.nm);
        }
        run(4096, 768, 768, 0, 0, OLD_BF16, "qkvo");
    }
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "OK", g_fail);
    return g_fail != 0;
}
