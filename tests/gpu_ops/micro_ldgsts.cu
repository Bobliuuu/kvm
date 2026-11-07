// Global -> shared (cp.async 16 B) throughput of the matmul_mma_bf16 tile loads, in isolation (no tensor-core work, no ldmatrix): how many bytes per
// clock per SM the access pattern sustains with every SM streaming the operand tiles of a 4096^3 bf16 GEMM, for K slabs of 32 and of 64.
//   nvcc -O2 -arch=sm_80 -std=c++17 micro_ldgsts.cu -o micro_ldgsts
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
__device__ __forceinline__ void cp16(unsigned dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src)); }
__device__ __forceinline__ void cp16z(unsigned dst, const void* src, bool ok) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(ok ? 16 : 0)); }
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// BK: k per slab. A is [M][K] (K-major rows of BK*2 bytes), B is [K][N] (MN-major rows of 512 bytes). NST stages in flight.
template <int BK, int NST, bool ZF = false>   // ZF: the zero-fill form of cp.async (src-size operand = in-range ? 16 : 0), as matmul_mma_bf16 issues on edge tiles
__global__ void __launch_bounds__(256, 1) k(const unsigned short* A, const unsigned short* B, int ld, int iters, int* sink) {
    extern __shared__ __align__(128) unsigned char sm[];
    const unsigned base = (unsigned)__cvta_generic_to_shared(sm);
    const int tid = threadIdx.x, m0 = (blockIdx.x % 32) * 128, n0 = (blockIdx.x / 32 % 16) * 256;
    constexpr int CA = 128 * (BK / 8), CB = BK * 32;   // chunks of A and B per slab
    auto issue = [&](int it) {
        const unsigned st = base + (it % NST) * (CA + CB) * 16;
        const int k0 = (it * BK) % 4096;
#pragma unroll
        for (int s = 0; s < CA / 256; s++) {
            const int q = tid + s * 256, row = q / (BK / 8), c = q % (BK / 8);
            if (ZF) cp16z(st + q * 16, A + (size_t)(m0 + row) * ld + k0 + c * 8, k0 + c * 8 < ld && m0 + row < ld); else cp16(st + q * 16, A + (size_t)(m0 + row) * ld + k0 + c * 8);
        }
#pragma unroll
        for (int s = 0; s < CB / 256; s++) {
            const int q = tid + s * 256, row = q / 32, c = q % 32;
            if (ZF) cp16z(st + CA * 16 + q * 16, B + (size_t)(k0 + row) * ld + n0 + c * 8, k0 + row < ld && n0 + c * 8 < ld); else cp16(st + CA * 16 + q * 16, B + (size_t)(k0 + row) * ld + n0 + c * 8);
        }
    };
    for (int p = 0; p < NST - 1; p++) { issue(p); commit(); }
    for (int it = 0; it < iters; it++) {
        wait<NST - 2>();
        __syncthreads();
        issue(it + NST - 1);
        commit();
    }
    wait<0>();
    if (tid == 0 && sink) sink[blockIdx.x] = 1;
}
int main() {
    const int ld = 4096;
    unsigned short *A, *B; cudaMalloc(&A, (size_t)ld * ld * 2); cudaMalloc(&B, (size_t)ld * ld * 2);
    cudaMemset(A, 1, (size_t)ld * ld * 2); cudaMemset(B, 1, (size_t)ld * ld * 2);
    cudaFuncSetAttribute(k<32, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    cudaFuncSetAttribute(k<64, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    cudaFuncSetAttribute(k<64, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 160 * 1024);
    cudaFuncSetAttribute(k<32, 6>, cudaFuncAttributeMaxDynamicSharedMemorySize, 144 * 1024);
    cudaFuncSetAttribute(k<32, 4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
    const int iters = 4096 / 32 * 4;
    struct V { const char* nm; int bk, nst, smem; void (*f)(const unsigned short*, const unsigned short*, int, int, int*); } vs[] = {
        {"BK32 NST4", 32, 4, 96 * 1024, k<32, 4>}, {"BK32 NST6", 32, 6, 144 * 1024, k<32, 6>}, {"BK64 NST2", 64, 2, 96 * 1024, k<64, 2>}, {"BK64 NST3", 64, 3, 144 * 1024, k<64, 3>},
        {"BK32 NST4 zero-fill form", 32, 4, 96 * 1024, k<32, 4, true>}};
    for (auto v : vs) {
        const int grid = 108, it = iters * 32 / v.bk;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        std::vector<float> ts;
        for (int r = 0; r < 5; r++) {
            cudaEventRecord(a); v.f<<<grid, 256, v.smem>>>(A, B, ld, it, nullptr); cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b); ts.push_back(ms);
        }
        std::sort(ts.begin(), ts.end());
        const double bytes = (double)grid * it * (128 * v.bk * 2 + v.bk * 256 * 2);
        const double bps = bytes / (ts[2] * 1e-3);
        printf("%-24s: %6.0f GB/s total, %5.1f B/clk/SM at 1.41 GHz, %.3f ms; equal to a %3.0f TFLOP/s GEMM at the 128x256 tile (85.3 FLOP per byte)\n", v.nm, bps * 1e-9,
               bps / grid / 1.41e9, ts[2], bps * 85.3 * 1e-12);
    }
    return 0;
}
