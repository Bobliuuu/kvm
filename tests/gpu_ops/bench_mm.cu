// Matmul-only benchmark (fast to compile: only matmul_tile is instantiated): the op library's matmul_tile against cuBLAS in ONE process,
// plus a correctness check against cuBLAS fp32 (full precision) per configuration.
//   -DMM_MINB=n  __launch_bounds__(256, n) on the benchmark kernel.   BENCH_REPS=n repetitions (median).
// A100 peaks: TF32 tensor 156 TFLOP/s.
#include <algorithm>
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
static double med(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }
template <typename F> static double time_ms(F f, int iters) {
    static const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 5;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) f();
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); for (int i = 0; i < iters; i++) f(); cudaEventRecord(b);
        CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); CK(cudaGetLastError()); t.push_back(ms / iters);
    }
    return med(t);
}
__global__ void fill(float* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
        p[i] = ((x & 0xffff) / 65536.f - 0.5f);
    }
}
static cublasHandle_t g_h;
// big: 0 = 64x64 tile, 1 = 128x128 (flag 32), 2 = 256x128 (flag 64); prec = 3xTF32
static void run(int M, int N, int K, int ta, int tb, int prec, int big, int bf = 0) {
    float *A, *B, *C, *R; CK(cudaMalloc(&A, (size_t)M * K * 4)); CK(cudaMalloc(&B, (size_t)K * N * 4)); CK(cudaMalloc(&C, (size_t)M * N * 4 + 16)); CK(cudaMalloc(&R, (size_t)M * N * 4));
    fill<<<1024, 256>>>(A, (size_t)M * K, 1); fill<<<1024, 256>>>(B, (size_t)K * N, 2);
    KvmBuf tab[3]{}; tab[0].p = A; tab[0].rows = ta ? K : M; tab[0].cols = ta ? M : K; tab[1].p = B; tab[1].rows = tb ? N : K; tab[1].cols = tb ? K : N; tab[2].p = C; tab[2].rows = M; tab[2].cols = N;
    for (auto& b : tab) b.ld = b.cols;
    KvmBuf* tab_d; CK(cudaMalloc(&tab_d, sizeof tab)); CK(cudaMemcpy(tab_d, tab, sizeof tab, cudaMemcpyHostToDevice));
    int32_t* err; CK(cudaMalloc(&err, 4)); CK(cudaMemset(err, 0, 4));
    const int bm = big == 2 ? 256 : big == 1 ? 128 : 64, bn = big ? 128 : 64, fl = ta | (tb << 1) | (prec << 4) | (bf << 7) | (big == 1 ? 32 : big == 2 ? 64 : 0) | (big && !prec && !bf && getenv("MM_RZ") ? 256 : 0);   // MM_RZ=1: flag 256, operands truncated like cuBLAS TF32
    const float alpha = 1.f; int32_t af; memcpy(&af, &alpha, 4);
    std::vector<int32_t> w; int nt = 0;
    for (int m0 = 0; m0 < M; m0 += bm) for (int n0 = 0; n0 < N; n0 += bn) { int32_t a[13] = {0, 1, 2, -1, -1, -1, m0, n0, M, N, K, fl, af}; w.insert(w.end(), a, a + 13); nt++; }
    int32_t* wd; CK(cudaMalloc(&wd, w.size() * 4)); CK(cudaMemcpy(wd, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    KvmCtx c{tab_d, 3, err};
    static bool attr = false; if (!attr) { CK(cudaFuncSetAttribute(k_mm, cudaFuncAttributeMaxDynamicSharedMemorySize, kvm_gpt_smem_bytes(64))); attr = true; }
    auto kvm = [&] { k_mm<<<nt, 256, kvm_gpt_smem_bytes(64)>>>(c, wd, 13); };
    kvm(); CK(cudaDeviceSynchronize());
    int e; CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
    // reference: cuBLAS full fp32 (no tf32)
    const float one = 1.f, zero = 0.f;
    cublasSetMathMode(g_h, CUBLAS_DEFAULT_MATH);
    auto cb = [&] { cublasSgemm(g_h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, tb ? K : N, A, ta ? M : K, &zero, R, N); };
    cb(); CK(cudaDeviceSynchronize());
    std::vector<float> hc((size_t)M * N), hr((size_t)M * N);
    CK(cudaMemcpy(hc.data(), C, hc.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hr.data(), R, hr.size() * 4, cudaMemcpyDeviceToHost));
    double me = 0, mx = 0; for (size_t i = 0; i < hc.size(); i++) { me = std::max(me, (double)fabsf(hc[i] - hr[i])); mx = std::max(mx, (double)fabsf(hr[i])); }
    const double tol = prec ? 2e-6 * sqrt((double)K) : bf ? 3e-2 : 6e-3;
    cublasSetMathMode(g_h, prec ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH);
    double flops = 2.0 * M * N * K, tk = time_ms(kvm, 10), tc = time_ms(cb, 10);
    printf("MM %-7s %c%c %5dx%5dx%5d tile %-7s kvm %7.1f cuBLAS %7.1f TFLOP/s  ratio %.3f  err_vs_fp32 %.1e %s dev_err %d\n", prec ? "3xTF32" : bf ? "bf16" : "TF32", ta ? 'T' : 'N', tb ? 'T' : 'N', M, N, K,
           big == 2 ? "256x128" : big == 1 ? "128x128" : "64x64", flops / tk * 1e-9, flops / tc * 1e-9, tc / tk, me / mx, me / mx < tol ? "ok" : "FAIL", e);
    fflush(stdout);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(R); cudaFree(tab_d); cudaFree(err); cudaFree(wd);
}
int main(int argc, char** argv) {
    cublasCreate(&g_h);
    const char* only = getenv("MM_ONLY");   // e.g. "NN4096" restricts shapes
    const int tiles[3] = {0, 1, 2};
    struct S { int M, N, K; } shapes[] = {{4096, 4096, 4096}, {2048, 2048, 2048}, {8192, 768, 768}, {4096, 3072, 768}, {1024, 1024, 1024}, {512, 512, 512}};
    if (getenv("MM_GPT162")) {   // the gpt_162m GEMMs (4096 tokens): forward NN, input-grad NT, weight-grad TN, each at 128x128 and 256x128
        struct G { int M, N, K; } g[] = {{4096, 2304, 768}, {4096, 768, 768}, {4096, 3072, 768}, {4096, 768, 3072}, {4096, 50304, 768}};
        for (auto s : g) for (int t = 1; t <= 2; t++) { run(s.M, s.N, s.K, 0, 0, 0, t); run(s.M, s.K, s.N, 0, 1, 0, t); run(s.K, s.N, s.M, 1, 0, 0, t); }
        return 0;
    }
    for (auto s : shapes) {
        if (only && (atoi(only) != s.M)) continue;
        for (int t : tiles) {
            if (t == 2 && getenv("MM_NO256")) continue;
            run(s.M, s.N, s.K, 0, 0, 0, t);
            if (s.M == 4096 && s.N == 4096) { run(s.M, s.N, s.K, 1, 0, 0, t); run(s.M, s.N, s.K, 0, 1, 0, t); }
        }
    }
    if (!getenv("MM_NOBF")) for (int t = 0; t < 3; t++) { run(4096, 4096, 4096, 0, 0, 0, t, 1); if (!getenv("MM_ONLY_NN")) { run(4096, 4096, 4096, 1, 0, 0, t, 1); run(4096, 4096, 4096, 0, 1, 0, t, 1); } }
    if (!getenv("MM_NOPREC")) for (int t = 0; t < 3; t++) run(4096, 4096, 4096, 0, 0, 1, t);
    return 0;
}
