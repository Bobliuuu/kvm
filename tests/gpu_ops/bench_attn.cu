// Attention-only benchmark of the causal ops (attn_fwd, attn_bwd_delta, attn_bwd_dq, attn_bwd_dkv), single-op kernels (no all-ops
// dispatcher, so it builds in seconds instead of minutes), one block per task, tasks ordered heaviest first. Correctness (batch 0,
// head 0 against the double reference) comes before every timing. Throughput counts causal 2*B*H*T^2*hd FLOP for the forward and 2.5x
// that for the backward, as bench_ops.cu and ref_torch.py do.
// Build: nvcc -O2 -arch=sm_80 -std=c++17 -lineinfo -I../../include -I. bench_attn.cu -o bench_attn
// Run:   ./bench_attn [B T H Hkv]   (default 4 1024 12 12 = the 162M GPT; hd is -DBENCH_HD, default 64)   BENCH_REPS=n repetitions.
#define KVM_HARNESS_NO_FULL
#ifndef BENCH_HD
#define BENCH_HD 64
#endif
#define KVM_GPT_MAX_HD BENCH_HD
#include <algorithm>
#include "harness.cuh"
#include "refs.hpp"

#define ONE_OP_KERNEL(NAME) \
    __global__ void __launch_bounds__(KVM_GPT_THREADS) k_##NAME(KvmCtx c, const int32_t* w, const int* off) { kvm_op_##NAME(c, w + off[blockIdx.x]); }
ONE_OP_KERNEL(attn_fwd) ONE_OP_KERNEL(attn_bwd_delta) ONE_OP_KERNEL(attn_bwd_dq) ONE_OP_KERNEL(attn_bwd_dkv)

struct Stage { int32_t* w; int* off; int n; KvmCtx c; };
static Stage stage(Dev& d, const Batch& b) {
    d.sync_table(); Stage s{}; s.n = (int)b.op.size();
    CK(cudaMalloc(&s.w, b.w.size() * 4 + 4)); CK(cudaMalloc(&s.off, s.n * 4));
    CK(cudaMemcpy(s.w, b.w.data(), b.w.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(s.off, b.off.data(), s.n * 4, cudaMemcpyHostToDevice));
    s.c = KvmCtx{d.tab_d, (int32_t)d.tab.size(), d.err};
    return s;
}
using Kernel = void (*)(KvmCtx, const int32_t*, const int*);
static void launch(Kernel k, const Stage& s, int smem) { k<<<s.n, KVM_GPT_THREADS, smem>>>(s.c, s.w, s.off); }

static std::vector<double> time_ms(const std::function<void()>& f, int iters, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) f();
    CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; i++) f();
        cudaEventRecord(b); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); CK(cudaGetLastError());
        t.push_back(ms / iters);
    }
    std::sort(t.begin(), t.end());
    return t;
}
static void report(const char* what, int tasks, const std::vector<double>& t, double flop) {
    double med = t[t.size() / 2];
    printf("BENCH  %-22s %5d tasks  %8.1f us [%.1f..%.1f]  %7.2f TFLOP/s\n", what, tasks, med * 1e3, t.front() * 1e3, t.back() * 1e3, flop / med * 1e-9);
}

int main(int argc, char** argv) {
    int B = 4, T = 1024, H = 12, Hkv = 12, HD = BENCH_HD;
    if (argc >= 5) { B = atoi(argv[1]); T = atoi(argv[2]); H = atoi(argv[3]); Hkv = atoi(argv[4]); }
    const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 7, R = B * T, RB = KVM_GPT_ATTN_ROWS;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s, %d SMs, %d MHz; B=%d T=%d H=%d Hkv=%d hd=%d, %d rows per task\n", pr.name, pr.multiProcessorCount, pr.clockRate / 1000, B, T, H, Hkv, HD, RB);
    Rng r(4); Dev dv;
    int q = dv.addf(R, H * HD), k = dv.addf(R, Hkv * HD), v = dv.addf(R, Hkv * HD), o = dv.addf(R, H * HD), lse = dv.addf(1, B * H * T), dO = dv.addf(R, H * HD);
    int delta = dv.addf(1, B * H * T), dq = dv.addf(R, H * HD), dk = dv.addf(R, Hkv * HD), dvv = dv.addf(R, Hkv * HD);
    auto hq = rnd(r, (size_t)R * H * HD), hk = rnd(r, (size_t)R * Hkv * HD), hv = rnd(r, (size_t)R * Hkv * HD), hdo = rnd(r, (size_t)R * H * HD);
    dv.up(q, hq); dv.up(k, hk); dv.up(v, hv); dv.up(dO, hdo);
    const float sc = 1.f / sqrtf((float)HD);
    Batch f, bd, bq, bkv;
    for (int q0 = ((T - 1) / RB) * RB; q0 >= 0; q0 -= RB) for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) {
        f.add(OP_attn_fwd, A().b(q).b(k).b(v).b(o).b(lse).i(b).i(h).i(q0).i(T).i(H).i(Hkv).i(HD).f(sc));
        bq.add(OP_attn_bwd_dq, A().b(q).b(k).b(v).b(dO).b(lse).b(delta).b(dq).i(b).i(h).i(q0).i(T).i(H).i(Hkv).i(HD).f(sc));
    }
    for (int k0 = 0; k0 < T; k0 += RB) for (int b = 0; b < B; b++) for (int h = 0; h < Hkv; h++)   // dkv: early key blocks are heaviest
        bkv.add(OP_attn_bwd_dkv, A().b(q).b(k).b(v).b(dO).b(lse).b(delta).b(dk).b(dvv).i(b).i(h).i(k0).i(T).i(H).i(HD).f(sc));
    const int drows = getenv("BENCH_DELTA_ROWS") ? atoi(getenv("BENCH_DELTA_ROWS")) : 128;   // rows per attn_bwd_delta task
    for (int r0 = 0; r0 < R; r0 += drows) bd.add(OP_attn_bwd_delta, A().b(o).b(dO).b(delta).i(r0).i(std::min(R, r0 + drows)).i(T).i(H).i(HD));
    const int smem = kvm_gpt_smem_bytes(HD);
    for (Kernel kk : {(Kernel)k_attn_fwd, (Kernel)k_attn_bwd_delta, (Kernel)k_attn_bwd_dq, (Kernel)k_attn_bwd_dkv})
        CK(cudaFuncSetAttribute(kk, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    auto sf = stage(dv, f), sd = stage(dv, bd), sq = stage(dv, bq), skv = stage(dv, bkv);
    launch(k_attn_fwd, sf, smem); launch(k_attn_bwd_delta, sd, smem); launch(k_attn_bwd_dq, sq, smem); launch(k_attn_bwd_dkv, skv, smem);
    CK(cudaDeviceSynchronize());
    int e = dv.geterr(); if (e) { printf("FAIL  device error %d\n", e); g_fail++; }
    {   // batch 0, head 0 (kv head 0) against the double reference; dk/dv only without GQA sharing
        auto go = dv.down(o), gdq = dv.down(dq), gdk = dv.down(dk), gdv = dv.down(dvv);
        D qq(T * HD), kk(T * HD), vv(T * HD), ddo(T * HD);
        for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) { qq[t * HD + d] = hq[(size_t)t * H * HD + d]; kk[t * HD + d] = hk[(size_t)t * Hkv * HD + d]; vv[t * HD + d] = hv[(size_t)t * Hkv * HD + d]; ddo[t * HD + d] = hdo[(size_t)t * H * HD + d]; }
        D ro, rl, rdq, rdk, rdv; attn_d(1, T, 1, 1, HD, sc, qq, kk, vv, ro, rl, &ddo, &rdq, &rdk, &rdv);
        auto rel = [&](const std::vector<float>& g, int ld, const D& ref) { double me = 0, mx = 0; for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) { me = std::max(me, fabs(g[(size_t)t * ld + d] - ref[t * HD + d])); mx = std::max(mx, fabs(ref[t * HD + d])); } return me / mx; };
        char dd[96]; double e1 = rel(go, H * HD, ro);
        snprintf(dd, sizeof dd, "relerr %.2e", e1); verdict("bench check: fwd o", e1 < 2e-5, dd);
        if (H == Hkv) {
            double e2 = rel(gdq, H * HD, rdq), e3 = rel(gdk, Hkv * HD, rdk), e4 = rel(gdv, Hkv * HD, rdv);
            snprintf(dd, sizeof dd, "relerr %.2e", e2); verdict("bench check: dq", e2 < 5e-5, dd);
            snprintf(dd, sizeof dd, "relerr %.2e", e3); verdict("bench check: dk", e3 < 5e-5, dd);
            snprintf(dd, sizeof dd, "relerr %.2e", e4); verdict("bench check: dv", e4 < 5e-5, dd);
        }
    }
    const double fl = 2.0 * B * H * (double)T * T * HD;
    report("fwd", sf.n, time_ms([&] { launch(k_attn_fwd, sf, smem); }, 5, reps), fl);
    report("bwd delta", sd.n, time_ms([&] { launch(k_attn_bwd_delta, sd, smem); }, 5, reps), 0);
    report("bwd dq", sq.n, time_ms([&] { launch(k_attn_bwd_dq, sq, smem); }, 5, reps), fl * 1.25);
    report("bwd dkv", skv.n, time_ms([&] { launch(k_attn_bwd_dkv, skv, smem); }, 5, reps), fl * 1.25);
    auto tb = time_ms([&] { launch(k_attn_bwd_delta, sd, smem); launch(k_attn_bwd_dq, sq, smem); launch(k_attn_bwd_dkv, skv, smem); }, 5, reps);
    report("bwd (delta+dq+dkv)", sd.n + sq.n + skv.n, tb, fl * 2.5);
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
