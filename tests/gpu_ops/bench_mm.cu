// Matmul-only benchmark (fast to compile: only the GEMM tiles are instantiated): the op library's matmul_tile, the #16 tile set
// (kvmgpt::j16::matmul_tile, kvm_ops_gpt.cuh) and cuBLAS in ONE process, on the same task words, plus a correctness check of each set
// against cuBLAS fp32 (full precision) per configuration.
// The tile A/B: per configuration, every rep runs each arm once, back to back (ours, #16 at 144 KiB, #16 at 96 KiB, cuBLAS), so a clock
// or power state of the round is shared by its arms; the "paired" ratio is the median over reps of the per-rep ratios.
//   -DMM_MINB=n  __launch_bounds__(256, n) on the benchmark kernels.   BENCH_REPS=n reps (default 7).
//   -DKVM_GPT_MMA_SMEM=147456 compiles the #16 set's 256 x 128 BK 32 x 3 tile, which it takes when the launch has 144 KiB; the set is also
//   timed at the launch its main ships with, 96 KiB (BK 16 x 4). Our tiles always launch with kvm_gpt_smem_bytes(64) (144 KiB).
//   MM_SHAPES=panel,gpt162,square (default all three)   MM_TILES="12" tile kinds (0 = 64x64, 1 = 128x128, 2 = 256x128)   MM_TR="NT,TN"
//   MM_RZ=1 adds the truncated-operand runs (flag 256, our set only)   MM_NOBF skips the bf16-operand runs (flag 128, our set only)
//   MM_NOPREC skips the 3xTF32 runs   MM_DIRECT=1 also launches the #16 256 x 128 tile as its own kernel (same task words)
//   MM_CUBLAS_ERR=1 prints cuBLAS TF32's own error against its fp32 result (how cuBLAS treats TF32 operands)
// A100 peaks: TF32 tensor 156 TFLOP/s.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cublas_v2.h>
#include "kvm_ctx_shim.cuh"
#include "kvm_ops_gpt.cuh"
#ifndef MM_MINB
#define MM_MINB 1
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void __launch_bounds__(256, MM_MINB) k_mm(KvmCtx c, const int32_t* w, int words) {
    kvmgpt::matmul_tile(c, w + (size_t)blockIdx.x * words);
}
__global__ void __launch_bounds__(256, MM_MINB) k_mm_j16(KvmCtx c, const int32_t* w, int words) {
    const int32_t* const a = w + (size_t)blockIdx.x * words;
    if (!kvmgpt::j16::matmul_tile(c, a)) kvmgpt::matmul_tile_ours(c, a);   // the tasks #16's set hands back run on ours, as before
}
// MM_DIRECT=1: the #16 256 x 128 fast tile launched as its own kernel (same task words), to separate the tile from the op-kernel context
template <bool TA, bool TB, int BK, int NST>
__global__ void __launch_bounds__(256, 1) k_direct(KvmCtx c, const int32_t* w, int words, const float* A, int lda, const float* B, int ldb) {
    const int32_t* a = w + (size_t)blockIdx.x * words;
    kvmgpt::j16::mm_tf32_tile<TA, TB, false, 4, 2, 64, BK, NST>(A, lda, B, ldb, a[6], a[7], a[8], a[9], a[10], c, a);
}
static double med(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
static int reps() { static const int r = getenv("BENCH_REPS") ? std::max(1, atoi(getenv("BENCH_REPS"))) : 7; return r; }
template <typename F> static double time_once(F f, int iters) {   // ms per call, one measurement
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a)); for (int i = 0; i < iters; i++) f(); CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); CK(cudaGetLastError());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / iters;
}
__global__ void fill(float* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
        p[i] = ((x & 0xffff) / 65536.f - 0.5f);
    }
}
static cublasHandle_t g_h;
static const int SMEM_OURS = kvm_gpt_smem_bytes(64), SMEM_J16_DEEP = 3 * (256 + 128) * 32 * 4, SMEM_J16_SHIP = 96 * 1024;
static int g_fails = 0;

// big: 0 = 64x64 tile, 1 = 128x128 (flag 32), 2 = 256x128 (flag 64); prec: 3xTF32 (flag 16); bf: bf16 operands (flag 128); rz: flag 256
static void run(const char* set, int M, int N, int K, int ta, int tb, int prec, int big, int bf = 0, int rz = 0) {
    if (getenv("MM_TILES") && !strchr(getenv("MM_TILES"), '0' + big)) return;
    if (getenv("MM_TR")) { const char tr[3] = {ta ? 'T' : 'N', tb ? 'T' : 'N', 0}; if (!strstr(getenv("MM_TR"), tr)) return; }
    float *A, *B, *C, *R; CK(cudaMalloc(&A, (size_t)M * K * 4)); CK(cudaMalloc(&B, (size_t)K * N * 4)); CK(cudaMalloc(&C, (size_t)M * N * 4 + 16)); CK(cudaMalloc(&R, (size_t)M * N * 4));
    fill<<<1024, 256>>>(A, (size_t)M * K, 1); fill<<<1024, 256>>>(B, (size_t)K * N, 2);
    KvmBuf tab[3]{}; tab[0].p = A; tab[0].rows = ta ? K : M; tab[0].cols = ta ? M : K; tab[1].p = B; tab[1].rows = tb ? N : K; tab[1].cols = tb ? K : N; tab[2].p = C; tab[2].rows = M; tab[2].cols = N;
    for (auto& b : tab) b.ld = b.cols;
    KvmBuf* tab_d; CK(cudaMalloc(&tab_d, sizeof tab)); CK(cudaMemcpy(tab_d, tab, sizeof tab, cudaMemcpyHostToDevice));
    int32_t* err; CK(cudaMalloc(&err, 4)); CK(cudaMemset(err, 0, 4));
    const int bm = big == 2 ? 256 : big == 1 ? 128 : 64, bn = big ? 128 : 64;
    const int fl = ta | (tb << 1) | (prec << 4) | (bf << 7) | (big == 1 ? 32 : big == 2 ? 64 : 0) | (rz ? 256 : 0);
    const float alpha = 1.f; int32_t af; memcpy(&af, &alpha, 4);
    std::vector<int32_t> w; int nt = 0;
    for (int m0 = 0; m0 < M; m0 += bm) for (int n0 = 0; n0 < N; n0 += bn) { int32_t a[13] = {0, 1, 2, -1, -1, -1, m0, n0, M, N, K, fl, af}; w.insert(w.end(), a, a + 13); nt++; }
    int32_t* wd; CK(cudaMalloc(&wd, w.size() * 4)); CK(cudaMemcpy(wd, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    KvmCtx c{tab_d, 3, err};
    // the #16 set has a form only for plain 1xTF32 big tiles; elsewhere its dispatch hands the task to ours, so it is not timed apart
    const bool ab = big && !prec && !bf && !rz;
#if KVM_GPT_MMA_SMEM >= 147456
    const bool deep_built = true;
#else
    const bool deep_built = false;
#endif
    const int lda_ = ta ? M : K, ldb_ = tb ? K : N;
    auto ours = [&] { k_mm<<<nt, 256, SMEM_OURS>>>(c, wd, 13); };
    auto j16_deep = [&] { k_mm_j16<<<nt, 256, SMEM_J16_DEEP>>>(c, wd, 13); };
    auto j16_ship = [&] { k_mm_j16<<<nt, 256, SMEM_J16_SHIP>>>(c, wd, 13); };
    auto direct = [&] {
        auto launch = [&](auto kern) { kern<<<nt, 256, SMEM_J16_DEEP>>>(c, wd, 13, A, lda_, B, ldb_); };
#if KVM_GPT_MMA_SMEM >= 147456
        if (!ta && !tb) launch(k_direct<false, false, 32, 3>); else if (ta && !tb) launch(k_direct<true, false, 32, 3>);
        else if (!ta && tb) launch(k_direct<false, true, 32, 3>); else launch(k_direct<true, true, 32, 3>);
#else
        if (!ta && !tb) launch(k_direct<false, false, 16, 4>); else if (ta && !tb) launch(k_direct<true, false, 16, 4>);
        else if (!ta && tb) launch(k_direct<false, true, 16, 4>); else launch(k_direct<true, true, 16, 4>);
#endif
    };
    const bool do_direct = ab && big == 2 && getenv("MM_DIRECT");
    // reference: cuBLAS full fp32 (no tf32)
    const float one = 1.f, zero = 0.f;
    cublasSetMathMode(g_h, CUBLAS_DEFAULT_MATH);
    auto ref = [&] { cublasSgemm(g_h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, tb ? K : N, A, ta ? M : K, &zero, R, N); };
    ref(); CK(cudaDeviceSynchronize());
    std::vector<float> hr((size_t)M * N), hc((size_t)M * N);
    CK(cudaMemcpy(hr.data(), R, hr.size() * 4, cudaMemcpyDeviceToHost));
    double mx = 0; for (float v : hr) mx = std::max(mx, (double)fabsf(v));
    const double tol = prec ? 2e-6 * sqrt((double)K) : bf ? 3e-2 : rz ? 1.2e-2 : 6e-3;
    auto check = [&](auto f, int& dev_err) {   // run once into C (zeroed first), return the max error relative to |ref|max
        CK(cudaMemset(C, 0, (size_t)M * N * 4)); CK(cudaMemset(err, 0, 4));
        f(); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(&dev_err, err, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hc.data(), C, hc.size() * 4, cudaMemcpyDeviceToHost));
        double me = 0; for (size_t i = 0; i < hc.size(); i++) me = std::max(me, (double)fabsf(hc[i] - hr[i]));
        return me / mx;
    };
    int e_o = 0, e_d = 0, e_s = 0, e_x = 0;
    const double err_o = check(ours, e_o), err_d = ab && deep_built ? check(j16_deep, e_d) : 0, err_s = ab ? check(j16_ship, e_s) : 0,
                 err_x = do_direct ? check(direct, e_x) : 0;
    auto ok = [&](double er, int de) { const bool g = er < tol && de == 0; if (!g) g_fails++; return g ? "ok" : "FAIL"; };
    cublasSetMathMode(g_h, prec ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH);
    double ce = 0;
    if (!prec && getenv("MM_CUBLAS_ERR")) {
        float* R2; CK(cudaMalloc(&R2, (size_t)M * N * 4));
        cublasSgemm(g_h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, tb ? K : N, A, ta ? M : K, &zero, R2, N);
        std::vector<float> h2((size_t)M * N); CK(cudaMemcpy(h2.data(), R2, h2.size() * 4, cudaMemcpyDeviceToHost)); cudaFree(R2);
        for (size_t i = 0; i < h2.size(); i++) ce = std::max(ce, (double)fabsf(h2[i] - hr[i]));
        ce /= mx;
    }
    auto cb = [&] { cublasSgemm(g_h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, tb ? K : N, A, ta ? M : K, &zero, R, N); };
    // warm each arm, then interleaved reps
    const int iters = 10;
    for (int i = 0; i < 3; i++) { ours(); if (ab) { if (deep_built) j16_deep(); j16_ship(); } if (do_direct) direct(); cb(); }
    CK(cudaDeviceSynchronize());
    std::vector<double> to, td, ts, tx, tc, pd, ps;
    for (int r = 0; r < reps(); r++) {
        to.push_back(time_once(ours, iters));
        if (ab && deep_built) { td.push_back(time_once(j16_deep, iters)); pd.push_back(to.back() / td.back()); }
        if (ab) { ts.push_back(time_once(j16_ship, iters)); ps.push_back(to.back() / ts.back()); }
        if (do_direct) tx.push_back(time_once(direct, iters));
        tc.push_back(time_once(cb, iters));
    }
    const double flops = 2.0 * M * N * K, mo = med(to), mc = med(tc);
    auto tf = [&](double ms) { return flops / ms * 1e-9; };
    const char* kind = prec ? "3xTF32" : bf ? "bf16" : rz ? "TF32rz" : "TF32";
    const char* tile = big == 2 ? "256x128" : big == 1 ? "128x128" : "64x64";
    printf("MM %-6s %-7s %c%c %5dx%5dx%5d tile %-7s cuBLAS %6.1f | ours %6.1f TFLOP/s (%.3f of cuBLAS) err %.1e %s dev_err %d", set, kind, ta ? 'T' : 'N',
           tb ? 'T' : 'N', M, N, K, tile, tf(mc), tf(mo), mc / mo, err_o, ok(err_o, e_o), e_o);
    if (ab && deep_built) printf(" | j16@144K %6.1f (%.3f) err %.1e %s, paired j16/ours %.3f", tf(med(td)), mc / med(td), err_d, ok(err_d, e_d), med(pd));
    if (ab) printf(" | j16@96K %6.1f (%.3f) err %.1e %s, paired j16/ours %.3f", tf(med(ts)), mc / med(ts), err_s, ok(err_s, e_s), med(ps));
    if (do_direct) printf(" | j16 direct %6.1f err %.1e %s", tf(med(tx)), err_x, ok(err_x, e_x));
    if (!prec && getenv("MM_CUBLAS_ERR")) printf(" | cublas_err %.1e", ce);
    printf("\n");
    fflush(stdout);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(R); cudaFree(tab_d); cudaFree(err); cudaFree(wd);
}
static bool want(const char* name) { const char* s = getenv("MM_SHAPES"); return !s || strstr(s, name); }
int main() {
    cublasCreate(&g_h);
    int dev = 0, optin = 0; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (optin < SMEM_OURS) { printf("this device opts into %d bytes of shared memory a block; our tiles launch with %d (an A100-class part): stopping\n", optin, SMEM_OURS); return 2; }
    {   // every kernel may use the full dynamic shared memory; then ~1 s of cuBLAS TF32 so clocks and power settle before timing
        const int dsm = std::max(SMEM_OURS, SMEM_J16_DEEP);
        void* ks[] = {(void*)k_mm, (void*)k_mm_j16,
#if KVM_GPT_MMA_SMEM >= 147456
                      (void*)k_direct<false, false, 32, 3>, (void*)k_direct<true, false, 32, 3>, (void*)k_direct<false, true, 32, 3>, (void*)k_direct<true, true, 32, 3>,
#else
                      (void*)k_direct<false, false, 16, 4>, (void*)k_direct<true, false, 16, 4>, (void*)k_direct<false, true, 16, 4>, (void*)k_direct<true, true, 16, 4>,
#endif
        };
        for (void* k : ks) CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, dsm));
        const int n = 4096; float *x, *y, *z; CK(cudaMalloc(&x, 4ull * n * n)); CK(cudaMalloc(&y, 4ull * n * n)); CK(cudaMalloc(&z, 4ull * n * n));
        fill<<<1024, 256>>>(x, (size_t)n * n, 3); fill<<<1024, 256>>>(y, (size_t)n * n, 4);
        cublasSetMathMode(g_h, CUBLAS_TF32_TENSOR_OP_MATH); const float one = 1.f, zero = 0.f;
        for (int i = 0; i < 1000; i++) cublasSgemm(g_h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, x, n, y, n, &zero, z, n);
        CK(cudaDeviceSynchronize()); cudaFree(x); cudaFree(y); cudaFree(z);
    }
    printf("tile A/B: ours = kvmgpt::matmul_tile at %d B of shared memory; j16 = kvmgpt::j16::matmul_tile (#16) at %d B%s and %d B; %d paired reps\n",
           SMEM_OURS, SMEM_J16_DEEP, KVM_GPT_MMA_SMEM >= 147456 ? "" : " (not built: -DKVM_GPT_MMA_SMEM=147456)", SMEM_J16_SHIP, reps());
    const bool rz = getenv("MM_RZ") != nullptr;
    if (want("gpt162")) {   // the gpt_162m GEMMs (4096 tokens): forward NN, input-grad NT, weight-grad TN, at 128x128 and 256x128
        struct G { int M, N, K; } g[] = {{4096, 2304, 768}, {4096, 768, 768}, {4096, 3072, 768}, {4096, 768, 3072}, {4096, 50304, 768}};
        for (auto s : g) for (int t = 1; t <= 2; t++) {
            run("gpt162", s.M, s.N, s.K, 0, 0, 0, t); run("gpt162", s.M, s.K, s.N, 0, 1, 0, t); run("gpt162", s.K, s.N, s.M, 1, 0, 0, t);
            if (rz) { run("gpt162", s.M, s.N, s.K, 0, 0, 0, t, 0, 1); run("gpt162", s.M, s.K, s.N, 0, 1, 0, t, 0, 1); run("gpt162", s.K, s.N, s.M, 1, 0, 0, t, 0, 1); }
        }
    }
    if (want("panel")) {   // the narrow panel rows' GEMMs (96 tokens; the MLPs at h72 / h256 / h1024): 64x64 tiles, the same code in both sets
        struct P { int M, N, K, ta, tb; } p[] = {{96, 72, 48, 0, 0}, {96, 72, 72, 0, 1}, {72, 72, 96, 1, 0}, {96, 256, 256, 0, 0}, {96, 256, 256, 0, 1}, {256, 256, 96, 1, 0},
                                                 {96, 1024, 1024, 0, 0}, {96, 1024, 1024, 0, 1}, {1024, 1024, 96, 1, 0}};
        for (auto s : p) run("panel", s.M, s.N, s.K, s.ta, s.tb, 0, 0);
    }
    if (want("square")) {
        struct S { int M, N, K; } shapes[] = {{4096, 4096, 4096}, {2048, 2048, 2048}, {8192, 768, 768}, {4096, 3072, 768}, {1024, 1024, 1024}, {512, 512, 512}};
        for (auto s : shapes) for (int t = 0; t < 3; t++) {
            run("square", s.M, s.N, s.K, 0, 0, 0, t);
            if (s.M == 4096 && s.N == 4096) { run("square", s.M, s.N, s.K, 1, 0, 0, t); run("square", s.M, s.N, s.K, 0, 1, 0, t); if (rz) run("square", s.M, s.N, s.K, 0, 0, 0, t, 0, 1); }
        }
        if (!getenv("MM_NOPREC")) for (int t = 0; t < 3; t++) run("square", 4096, 4096, 4096, 0, 0, 1, t);
        if (!getenv("MM_NOBF")) for (int t = 0; t < 3; t++) { run("square", 4096, 4096, 4096, 0, 0, 0, t, 1); run("square", 4096, 4096, 4096, 1, 0, 0, t, 1); run("square", 4096, 4096, 4096, 0, 1, 0, t, 1); }
    }
    printf("%s: %d configuration(s) failed their error bound\n", g_fails ? "FAIL" : "PASS", g_fails);
    return g_fails ? 1 : 0;
}
