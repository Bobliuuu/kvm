// sm_90a TMA + wgmma GEMM tiles (include/kvm_hopper.cuh) through the real op entry points (matmul_tile, matmul_mma_bf16).
//   correctness: every transpose combination x tile kind (64x64, 128x128 flag 32, 256x128 flag 64; bf16 128x256) x epilogue mix x ragged
//                and aligned shapes x two launch sizes (the executor's 96 KiB sm_80 launch and the 200 KiB sm_90 launch), against a float64 CPU
//                reference of the same (tf32-truncated / bf16) inputs, and against the sm_80 mma.sync path of the same binary (kvm_tma_ready
//                cleared). The tile counters prove the Hopper path ran (a silent fallback fails the test).
//   bench (--bench): TFLOP/s of the Hopper tile, the sm_80 tile on the same GPU and cuBLAS (TF32 / bf16, fp32 accumulate), one CTA per tile.
// Build (sm_90a only): nvcc -O3 -arch=sm_90a -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -I../../include -I. test_hopper.cu -lcublas -o test_hopper
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <cublas_v2.h>
#ifndef KVM_GPT_MAX_HD
#define KVM_GPT_MAX_HD 64
#endif
#define KVM_HARNESS_NO_FULL
#include "harness.cuh"
#include "kvm_ops_extra.cuh"

__global__ void __launch_bounds__(256, 1) k_mm(KvmCtx c, const int32_t* w, int which) {
    const int32_t* a = w + 13 * blockIdx.x;
    if (which == 0) kvmgpt::matmul_tile(c, a); else kvmextra::matmul_mma_bf16(c, a);
}
static void set_ready(int on) { CK(cudaMemcpyToSymbol(kvm_tma_ready, &on, 4)); }
static unsigned long long counts(int i) { unsigned long long c[4]; kvm_hopper_counts(c, true); return c[i]; }

static float tf32t(float x) { uint32_t u; memcpy(&u, &x, 4); u &= 0xffffe000u; float y; memcpy(&y, &u, 4); return y; }   // what tf32 wgmma reads
static float bf16v(float x) { uint32_t u; memcpy(&u, &x, 4); u = (u + 0x7fffu + ((u >> 16) & 1)) & 0xffff0000u; float y; memcpy(&y, &u, 4); return y; }
static double gelu_d(double x) { return 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); }

struct Case { int M, N, K, fl; bool bias, res, pre, gelu, accum; };

// Launch every tile of C with one CTA each; returns the device error word.
static int launch(Dev& d, int which, const std::vector<int32_t>& w, int nt, int smem) {
    d.sync_table();
    int32_t* wd; CK(cudaMalloc(&wd, w.size() * 4)); CK(cudaMemcpy(wd, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
    KvmCtx c{d.tab_d, (int32_t)d.tab.size(), d.err};
    k_mm<<<nt, 256, smem>>>(c, wd, which);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaFree(wd);
    return d.geterr();
}
// One correctness case of matmul_tile (which 0, fp32) or matmul_mma_bf16 (which 1; ob: bf16 out/res/pre storage).
static void run_case(Rng& rng, int which, const Case& cs, bool ob, int smem) {
    const int M = cs.M, N = cs.N, K = cs.K, fl = cs.fl;
    const bool ta = fl & 1, tb = fl & 2;
    const int bm = which ? 128 : (fl & 32) ? 128 : (fl & 64) ? 256 : 64, bn = which ? 256 : (fl & 32) ? 128 : (fl & 64) ? 128 : 64;
    Dev d;
    const int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
    auto ha = rnd(rng, (size_t)ar * ac), hb = rnd(rng, (size_t)br * bc), hbias = rnd(rng, N), hres = rnd(rng, (size_t)M * N), hout0 = rnd(rng, (size_t)M * N);
    if (which) { for (auto* v : {&ha, &hb}) for (auto& x : *v) x = bf16v(x); if (ob) for (auto* v : {&hres, &hout0}) for (auto& x : *v) x = bf16v(x); }
    const int lda = ac + (which ? 8 : 4) * (rng.u32() & 1), ldb = bc + (which ? 8 : 4) * (rng.u32() & 1);   // padded rows half the time
    const int ia = which ? d.addb(ar, ac, lda) : d.addf(ar, ac, lda), ib = which ? d.addb(br, bc, ldb) : d.addf(br, bc, ldb);
    const bool sb = which && ob;
    const int io = sb ? d.addb(M, N) : d.addf(M, N), ibias = cs.bias ? d.addf(1, N) : -1, ires = cs.res ? (sb ? d.addb(M, N) : d.addf(M, N)) : -1,
              ipre = cs.pre ? (sb ? d.addb(M, N) : d.addf(M, N)) : -1;
    if (which) { d.upb(ia, ha); d.upb(ib, hb); } else { d.up(ia, ha); d.up(ib, hb); }
    if (cs.bias) d.up(ibias, hbias);
    if (cs.res) { if (sb) d.upb(ires, hres); else d.up(ires, hres); }
    const float alpha = 0.75f;
    int32_t af; memcpy(&af, &alpha, 4);
    std::vector<int32_t> w; int nt = 0;
    const int f = fl | (cs.accum ? 4 : 0) | (cs.gelu ? 8 : 0);
    for (int m0 = 0; m0 < M; m0 += bm) for (int n0 = 0; n0 < N; n0 += bn) {
        int32_t a[13] = {ia, ib, io, ibias, ires, ipre, m0, n0, M, N, K, f, af}; w.insert(w.end(), a, a + 13); nt++;
    }
    // float64 reference on the values the tensor cores read
    std::vector<double> rout((size_t)M * N), rpre((size_t)M * N);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double s = 0;
        for (int k = 0; k < K; k++) {
            float x = ta ? ha[(size_t)k * ac + m] : ha[(size_t)m * ac + k], y = tb ? hb[(size_t)n * bc + k] : hb[(size_t)k * bc + n];
            if (!which) { x = tf32t(x); y = tf32t(y); }
            s += (double)x * y;
        }
        double v = s * alpha + (cs.bias ? hbias[n] : 0);
        rpre[(size_t)m * N + n] = v;
        if (cs.gelu) v = gelu_d(v);
        if (cs.res) v += hres[(size_t)m * N + n];
        if (cs.accum) v += hout0[(size_t)m * N + n];
        rout[(size_t)m * N + n] = v;
    }
    char name[160];
    snprintf(name, sizeof name, "%s %dx%dx%d fl=%d%s%s%s%s%s%s smem=%dK", which ? "mma_bf16" : "matmul_tile", M, N, K, fl, cs.bias ? " bias" : "", cs.res ? " res" : "",
             cs.pre ? " pre" : "", cs.gelu ? " gelu" : "", cs.accum ? " accum" : "", sb ? " bf16-out" : "", smem / 1024);
    std::vector<float> got[2], gpre[2];
    unsigned long long taken[2];
    for (int path = 0; path < 2; path++) {   // 0: Hopper, 1: sm_80 tile (same binary, Hopper disarmed)
        set_ready(path == 0);
        if (cs.accum) { if (sb) d.upb(io, hout0); else d.up(io, hout0); }
        counts(0);
        const int e = launch(d, which, w, nt, smem);
        taken[path] = counts(which);
        check_true(std::string(name) + (path ? " [sm80] err" : " [hopper] err"), e == 0, e ? "device error" : "");
        got[path] = sb ? d.downb(io) : d.down(io);
        if (cs.pre) gpre[path] = sb ? d.downb(ipre) : d.down(ipre);
    }
    set_ready(1);
    const double tol = which ? (sb ? 8e-3 : 2e-5) : 4e-3;   // tf32: matmul_tile's 1xTF32 bound; bf16: exact products, fp32 sums (bf16 out: one ulp)
    check(std::string(name) + " hopper vs f64", got[0], rout, tol);
    check(std::string(name) + " sm80 vs f64", got[1], rout, which ? tol : 4e-3);
    if (cs.pre) check(std::string(name) + " hopper pre vs f64", gpre[0], rpre, tol);
    // The Hopper path must run whenever its pipeline fits the launch (same rule as kvm_hopper.cuh: launch - 2048 bytes usable, at least two
    // stages plus the double-buffered transposed copies); otherwise it must decline for every tile.
    const long avail = (long)smem - 2048;
    const long need = which ? 2L * (128 + 256) * 32 * 2 : 2L * (bm + bn) * 128 + 2L * ((ta ? bm : 0) + (!tb ? bn : 0)) * 128;
    const bool fits = need <= avail && !(!which && bm == 64 && (ta || !tb));   // 64x64 tf32: Hopper only for K-major A and B
    char det[96]; snprintf(det, sizeof det, "%llu of %d tiles (expected %s)", taken[0], nt, fits ? "all" : "none: pipeline does not fit");
    check_true(std::string(name) + " hopper path taken", taken[0] == (fits ? (unsigned long long)nt : 0ull), det);
    check_true(std::string(name) + " sm80 path when disarmed", taken[1] == 0, "");
}

static void correctness(int smem) {
    Rng rng(90);
    const int shapes[][3] = {{64, 64, 32}, {64, 64, 768}, {200, 136, 100}, {130, 70, 45}, {256, 256, 256}, {384, 260, 515}, {1000, 300, 129}};
    for (auto& s : shapes)
        for (int t = 0; t < 4; t++)
            for (int kind : {0, 32, 64}) {
                const int r = rng.u32();
                Case cs{s[0], s[1], s[2], t | kind, bool(r & 1), bool(r & 2), bool(r & 4), bool(r & 8), bool(r & 16)};
                if (s[0] % 4 || s[1] % 4 || s[2] % 4) { cs.M = s[0] & ~3; cs.N = s[1] & ~3; cs.K = s[2] & ~3; if (!cs.K) cs.K = 4; }   // TMA: 16-byte rows
                run_case(rng, 0, cs, false, smem);
            }
    const int bshapes[][3] = {{128, 256, 32}, {128, 256, 768}, {256, 512, 64}, {200, 264, 104}, {136, 520, 40}};
    for (auto& s : bshapes)
        for (int t = 0; t < 4; t++)
            for (int ob = 0; ob < 2; ob++) {
                const int r = rng.u32();
                Case cs{s[0], s[1], s[2], t, bool(r & 1), bool(r & 2), bool(r & 4), bool(r & 8), bool(r & 16)};
                run_case(rng, 1, cs, ob, smem);
            }
}

// ---- bench --------------------------------------------------------------------------------------------------------------------------------
static double time_ms(const std::function<void()>& f, int reps) {
    f(); CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> t;
    for (int r = 0; r < reps; r++) { cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); t.push_back(ms); }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
static void bench(int smem) {
    cublasHandle_t h; cublasCreate(&h);
    struct S { const char* what; int M, N, K, fl, which; };
    const S ss[] = {{"4096^3 NN", 4096, 4096, 4096, 0, 0}, {"4096^3 NT", 4096, 4096, 4096, 2, 0}, {"4096^3 TN", 4096, 4096, 4096, 1, 0},
                    {"4096^3 NN big", 4096, 4096, 4096, 32, 0}, {"4096^3 NT big", 4096, 4096, 4096, 34, 0}, {"4096^3 TN big", 4096, 4096, 4096, 33, 0}, {"4096^3 TT big", 4096, 4096, 4096, 35, 0}, {"4096^3 NN huge", 4096, 4096, 4096, 64, 0},
                    {"4096^3 TN huge", 4096, 4096, 4096, 65, 0}, {"4096^3 TT huge", 4096, 4096, 4096, 67, 0},
                    {"4096^3 NT huge", 4096, 4096, 4096, 66, 0}, {"qkv 4096x2304x768 NN", 4096, 2304, 768, 0, 0}, {"fc1 4096x3072x768 NN", 4096, 3072, 768, 0, 0},
                    {"bf16 4096^3 NN", 4096, 4096, 4096, 0, 1}, {"bf16 4096^3 NT", 4096, 4096, 4096, 2, 1}, {"bf16 4096^3 TN", 4096, 4096, 4096, 1, 1},
                    {"bf16 4096^3 TT", 4096, 4096, 4096, 3, 1}, {"bf16 logits 4096x50304x768 NT", 4096, 50304, 768, 2, 1}};
    printf("%-34s %10s %10s %10s %8s\n", "case", "hopper", "sm80", "cuBLAS", "hop/cuB");
    for (auto& s : ss) {
        Dev d;
        const bool ta = s.fl & 1, tb = s.fl & 2;
        const int ar = ta ? s.K : s.M, ac = ta ? s.M : s.K, br = tb ? s.N : s.K, bc = tb ? s.K : s.N;
        Rng rng(7);
        const int ia = s.which ? d.addb(ar, ac) : d.addf(ar, ac), ib = s.which ? d.addb(br, bc) : d.addf(br, bc), io = d.addf(s.M, s.N);
        if (s.which) { auto x = rnd(rng, (size_t)ar * ac); for (auto& v : x) v = bf16v(v); d.upb(ia, x); auto y = rnd(rng, (size_t)br * bc); for (auto& v : y) v = bf16v(v); d.upb(ib, y); }
        else { d.up(ia, rnd(rng, (size_t)ar * ac)); d.up(ib, rnd(rng, (size_t)br * bc)); }
        const int bm = s.which ? 128 : (s.fl & 32) ? 128 : (s.fl & 64) ? 256 : 64, bn = s.which ? 256 : (s.fl & 96) ? 128 : 64;
        float alpha = 1.f; int32_t af; memcpy(&af, &alpha, 4);
        std::vector<int32_t> w; int nt = 0;
        const bool same = getenv("HOP_SAME") != nullptr;   // diagnostic: every CTA computes tile (0,0) (operands stay in L2: the compute ceiling)
        for (int m0 = 0; m0 < s.M; m0 += bm) for (int n0 = 0; n0 < s.N; n0 += bn) { int32_t a[13] = {ia, ib, io, -1, -1, -1, same ? 0 : m0, same ? 0 : n0, s.M, s.N, s.K, s.fl, af}; w.insert(w.end(), a, a + 13); nt++; }
        d.sync_table();
        int32_t* wd; CK(cudaMalloc(&wd, w.size() * 4)); CK(cudaMemcpy(wd, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
        KvmCtx c{d.tab_d, (int32_t)d.tab.size(), d.err};
        auto kv = [&] { k_mm<<<nt, 256, smem>>>(c, wd, s.which); };
        set_ready(1); counts(0);
        const double th = time_ms(kv, 7);
        const unsigned long long taken = counts(s.which);
        set_ready(0);
        const double t8 = time_ms(kv, 5);
        set_ready(1);
        // cuBLAS, column-major view: C^T (N x M) = op(B)^T op(A)^T
        const float one = 1.f, zero = 0.f;
        cublasSetMathMode(h, s.which ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH);
        const cudaDataType ty = s.which ? CUDA_R_16BF : CUDA_R_32F;
        auto cb = [&] {
            cublasGemmEx(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, s.N, s.M, s.K, &one, d.tab[ib].p, ty, d.tab[ib].ld, d.tab[ia].p, ty,
                         d.tab[ia].ld, &zero, d.tab[io].p, CUDA_R_32F, s.N, s.which ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        };
        const double tc = time_ms(cb, 7);
        const double fl = 2.0 * s.M * s.N * s.K / 1e9;
        printf("%-34s %10.1f %10.1f %10.1f %8.2f   (TFLOP/s; hopper tiles %llu/%d, err %d)\n", s.what, fl / th, fl / t8, fl / tc, tc / th, taken, nt, d.geterr());
        cudaFree(wd);
    }
    cublasDestroy(h);
}

int main(int argc, char** argv) {
    int dev; cudaDeviceProp p; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&p, dev));
    printf("device %s sm_%d%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
    if (!kvm_hopper_host_init()) { printf("FAIL  kvm_hopper_host_init (needs sm_90 and the driver's cuTensorMapEncodeTiled)\n"); return 1; }
    const int big = KVM_HOP_SMEM_BYTES, small = kvm_gpt_smem_bytes(KVM_GPT_MAX_HD);
    CK(cudaFuncSetAttribute(k_mm, cudaFuncAttributeMaxDynamicSharedMemorySize, big));
    if (argc > 1 && !strcmp(argv[1], "--bench")) { bench(argc > 2 ? atoi(argv[2]) * 1024 : big); return 0; }
    g_quiet = argc > 1 && !strcmp(argv[1], "--quiet");
    correctness(small);
    correctness(big);
    printf("test_hopper: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
