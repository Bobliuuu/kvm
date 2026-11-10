// Single-instantiation benchmark of the 128x128 TF32 tile (mm_tile_big) against cuBLAS TF32, in one process. It compiles in well under a
// minute (bench_mm.cu instantiates every flag/transpose combination of matmul_tile), so it is the loop for tile-kernel changes.
//   nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. [-DTILE_TA=1 -DTILE_TB=1] [-DMM_MINB=2] bench_tile.cu -lcublas
//   ./bench_tile [M N K]     (default 4096 4096 4096; TILE_TA/TILE_TB select the operand layouts, NN by default)
// Prints TFLOP/s for the tile kernel and cuBLAS and the max error against cuBLAS fp32.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>
#include "kvm_ctx_shim.cuh"
#include "kvm_ops_gpt.cuh"
#ifndef MM_MINB
#define MM_MINB 2
#endif
#ifndef TILE_WM
#define TILE_WM 2   // warps along m x along n (8 warps), each owning a 64 x TILE_WTN sub-tile: tile = 64*WM x WTN*WN
#define TILE_WN 4
#define TILE_WTN 32
#endif
#define TILE_BM (64 * TILE_WM)
#define TILE_BN (TILE_WTN * TILE_WN)
#ifndef TILE_RZ
#define TILE_RZ 0   // 1: operands truncated by the tensor core (matmul_tile flag 256) instead of rounded to tf32
#endif
#ifndef TILE_TA
#define TILE_TA 0
#endif
#ifndef TILE_TB
#define TILE_TB 0
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void __launch_bounds__(256, MM_MINB) k_tile(const float* A, int lda, const float* B, int ldb, float* C, int M, int N, int K) {
    const int tn = (N + TILE_BN - 1) / TILE_BN, m0 = (blockIdx.x / tn) * TILE_BM, n0 = (blockIdx.x % tn) * TILE_BN;
    kvmgpt::mm_tile_big<TILE_TA, TILE_TB, false, TILE_WM, TILE_WN, TILE_WTN, false, TILE_RZ>(A, lda, B, ldb, C, N, nullptr, nullptr, 0, nullptr, 0, m0, n0, M, N, K, 1.f, false, false);
}
__global__ void fill(float* p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13;
        p[i] = ((x & 0xffff) / 65536.f - 0.5f);
    }
}
template <typename F> static double best_ms(F f, int iters, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) f();
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a); for (int i = 0; i < iters; i++) f(); cudaEventRecord(b);
        CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); CK(cudaGetLastError()); t.push_back(ms / iters);
    }
    std::sort(t.begin(), t.end());
    printf("   (min %.4f med %.4f max %.4f ms)\n", t[0], t[t.size() / 2], t.back());
    return t[0];
}
int main(int argc, char** argv) {
    const int M = argc > 3 ? atoi(argv[1]) : 4096, N = argc > 3 ? atoi(argv[2]) : 4096, K = argc > 3 ? atoi(argv[3]) : 4096;
    float *A, *B, *C, *R;
    CK(cudaMalloc(&A, (size_t)M * K * 4)); CK(cudaMalloc(&B, (size_t)K * N * 4)); CK(cudaMalloc(&C, (size_t)M * N * 4)); CK(cudaMalloc(&R, (size_t)M * N * 4));
    fill<<<1024, 256>>>(A, (size_t)M * K, 1); fill<<<1024, 256>>>(B, (size_t)K * N, 2);
    const int lda = TILE_TA ? M : K, ldb = TILE_TB ? K : N, smem = std::max(kvm_gpt_smem_bytes(64), (TILE_BM + TILE_BN) * KVM_GPT_BIG_BK * KVM_GPT_BIG_NST * 4);
    CK(cudaFuncSetAttribute(k_tile, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    const int nt = ((M + TILE_BM - 1) / TILE_BM) * ((N + TILE_BN - 1) / TILE_BN);
    auto tile = [&] { k_tile<<<nt, 256, smem>>>(A, lda, B, ldb, C, M, N, K); };
    cublasHandle_t h; cublasCreate(&h);
    const float one = 1.f, zero = 0.f;
    auto cb = [&] { cublasSgemm(h, TILE_TB ? CUBLAS_OP_T : CUBLAS_OP_N, TILE_TA ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &one, B, ldb, A, lda, &zero, R, N); };
    tile(); cublasSetMathMode(h, CUBLAS_DEFAULT_MATH); cb(); CK(cudaDeviceSynchronize());
    std::vector<float> hc((size_t)M * N), hr((size_t)M * N);
    CK(cudaMemcpy(hc.data(), C, hc.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hr.data(), R, hr.size() * 4, cudaMemcpyDeviceToHost));
    double me = 0, mx = 0;
    for (size_t i = 0; i < hc.size(); i++) { me = std::max(me, (double)fabsf(hc[i] - hr[i])); mx = std::max(mx, (double)fabsf(hr[i])); }
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    const double flops = 2.0 * M * N * K;
    printf("tile %dx%d %c%c %dx%dx%d MINB=%d smem %d\n", TILE_BM, TILE_BN, TILE_TA ? 'T' : 'N', TILE_TB ? 'T' : 'N', M, N, K, MM_MINB, smem);
    const double tk = best_ms(tile, 10, 7), tc = best_ms(cb, 10, 7);
    printf("TILE %.1f TFLOP/s   cuBLAS-TF32 %.1f TFLOP/s   ratio %.3f   err_vs_fp32 %.1e %s\n", flops / tk * 1e-9, flops / tc * 1e-9, tc / tk, me / mx, me / mx < 6e-3 ? "ok" : "FAIL");
    return 0;
}
