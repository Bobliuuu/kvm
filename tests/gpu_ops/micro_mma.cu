// Peak probes for the mma.sync m16n8k16 bf16 path on one A100: (1) HMMA only, (2) HMMA + ldmatrix from shared memory, per SM with 8 warps.
//   nvcc -O2 -arch=sm_80 -std=c++17 micro_mma.cu -o micro_mma
#include <cstdio>
#include <cuda_runtime.h>
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
template <int MODE>   // 0: HMMA only; 1: + 8 ldmatrix.x4 per 32 HMMA (the 64x64 warp tile's per-k16 load), results consumed
__global__ void __launch_bounds__(256, 1) k(float* out, int iters, unsigned seed) {
    extern __shared__ __align__(128) unsigned char sm[];
    float acc[4][8][4];
    unsigned af[4][4], bf[8][2];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    for (int i = 0; i < 4; i++) for (int e = 0; e < 4; e++) af[i][e] = seed + threadIdx.x + i * 7 + e;
    for (int j = 0; j < 8; j++) for (int e = 0; e < 2; e++) bf[j][e] = seed * 3 + threadIdx.x + j * 5 + e;
    const unsigned base = (unsigned)__cvta_generic_to_shared(sm) + (threadIdx.x & 31) * 16;
    for (int it = 0; it < iters; it++) {
        if (MODE == 1) {
            unsigned r[4];
            for (int q = 0; q < 4; q++) { ldsm_x4(r, base + q * 512 + (it & 3) * 2048); af[q][0] = r[0]; af[q][1] = r[1]; af[q][2] = r[2]; af[q][3] = r[3]; }
            for (int q = 0; q < 4; q++) { ldsm_x4(r, base + 8192 + q * 512 + (it & 3) * 2048); bf[2 * q][0] = r[0]; bf[2 * q][1] = r[1]; bf[2 * q + 1][0] = r[2]; bf[2 * q + 1][1] = r[3]; }
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) mma_bf16(acc[i][j], af[i], bf[j][0], bf[j][1]);
    }
    float s = 0;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) for (int e = 0; e < 4; e++) s += acc[i][j][e];
    out[blockIdx.x * 256 + threadIdx.x] = s;
}
int main() {
    float* out; cudaMalloc(&out, 108 * 256 * 4 * 4);
    cudaFuncSetAttribute(k<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 72 * 1024);
    for (int mode = 0; mode < 2; mode++) {
        const int iters = 20000;
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        for (int rep = 0; rep < 2; rep++) {
            cudaEventRecord(a);
            if (mode == 0) k<0><<<108, 256, 72 * 1024>>>(out, iters, 1); else k<1><<<108, 256, 72 * 1024>>>(out, iters, 1);
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            double flops = 108.0 * 8 * iters * 32 * 2.0 * 16 * 8 * 16;
            if (rep) printf("mode %d: %.1f TFLOP/s (%.3f ms)\n", mode, flops / ms * 1e-9, ms);
        }
    }
    return 0;
}
