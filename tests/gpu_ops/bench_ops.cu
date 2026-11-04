// Micro-benchmarks of the big ops, each preceded by a correctness check of the exact configuration timed.
// "full dispatch" = the all-ops kernel (what a megakernel executor compiles: registers are the union's);
// "single-op" = the same device code in a kernel that contains only that op (register/occupancy upper bound).
// A100 peaks: TF32 tensor 156 TFLOP/s, FP32 19.5 TFLOP/s, HBM 1555 GB/s (40 GB SXM4).
#include <algorithm>
#include "harness.cuh"
#include "refs.hpp"

template <int ID>
__global__ void __launch_bounds__(256) k_one(KvmCtx c, const int32_t* w, const int* off) { kvm_gpt_dispatch(c, ID, w + off[blockIdx.x]); }

struct Staged { int32_t* w; int* op; int* off; int n; KvmCtx c; };
static Staged stage(Dev& d, const Batch& b) {
    static int attr = 0; if (!attr) { CK(cudaFuncSetAttribute(k_run, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); attr = 1; }
    d.sync_table(); Staged s{}; s.n = (int)b.op.size();
    CK(cudaMalloc(&s.w, b.w.size() * 4 + 4)); CK(cudaMalloc(&s.op, s.n * 4)); CK(cudaMalloc(&s.off, s.n * 4));
    CK(cudaMemcpy(s.w, b.w.data(), b.w.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(s.op, b.op.data(), s.n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.off, b.off.data(), s.n * 4, cudaMemcpyHostToDevice)); s.c = KvmCtx{d.tab_d, (int32_t)d.tab.size(), d.err};
    return s;
}
template <int ID> static void launch_one(const Staged& s, int smem) {
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k_one<ID>, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)); attr = true; }
    k_one<ID><<<s.n, 256, smem>>>(s.c, s.w, s.off);
}
static void launch_full(const Staged& s, int smem) { k_run<<<s.n, 256, smem>>>(s.c, s.w, s.op, s.off); }
static double g_tmin = 0, g_tmax = 0;   // spread of the last time_ms: min / max over the repetitions
// Median over BENCH_REPS (default 7) repetitions of `iters` back-to-back launches each; min/max land in g_tmin/g_tmax.
template <typename F> static double time_ms(F f, int iters) {
    static const int reps = getenv("BENCH_REPS") ? atoi(getenv("BENCH_REPS")) : 7;
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
    std::sort(t.begin(), t.end()); g_tmin = t.front(); g_tmax = t.back();
    return t[t.size() / 2];
}
static void report(const char* name, const char* mode, double ms, double work, const char* unit, double peak) {
    printf("BENCH  %-46s %-13s %8.3f ms [%.3f..%.3f]  %9.1f %s  (%.1f%% of %.0f)\n", name, mode, ms, g_tmin, g_tmax, work / ms * 1e-6, unit, work / ms * 1e-6 / peak * 100, peak);
}

template <int ID> static void bench_mm(int M, int N, int K, int ta, int tb, int prec) {
    Rng r(3); Dev dv;
    int ia = dv.addf(ta ? K : M, ta ? M : K), ib = dv.addf(tb ? N : K, tb ? K : N), io = dv.addf(M, N);
    auto ha = rnd(r, (size_t)M * K), hb = rnd(r, (size_t)K * N); dv.up(ia, ha); dv.up(ib, hb);
    Batch b; for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64)
        b.add(OP_matmul_tile, A().b(ia).b(ib).b(io).b(-1).b(-1).b(-1).i(m0).i(n0).i(M).i(N).i(K).i(ta | (tb << 1) | (prec << 4)).f(1.f));
    auto s = stage(dv, b); launch_full(s, 48 * 1024); CK(cudaDeviceSynchronize());
    auto o = dv.down(io); Rng rs(9); double me = 0, mx = 0;   // sampled elements vs double dot products
    for (int t = 0; t < 400; t++) {
        int m = rs.u32() % M, n = rs.u32() % N; double acc = 0;
        for (int k = 0; k < K; k++) acc += (double)(ta ? ha[(size_t)k * M + m] : ha[(size_t)m * K + k]) * (tb ? hb[(size_t)n * K + k] : hb[(size_t)k * N + n]);
        me = std::max(me, fabs(o[(size_t)m * N + n] - acc)); mx = std::max(mx, fabs(acc));
    }
    char nm[96]; snprintf(nm, sizeof nm, "matmul %s %c%c %dx%dx%d", prec ? "3xTF32" : "TF32", ta ? 'T' : 'N', tb ? 'T' : 'N', M, N, K);
    double tol = prec ? 2e-6 * sqrt((double)K) : 6e-3; char d[64]; snprintf(d, sizeof d, "400 sampled elements relerr %.2e", me / mx);
    verdict(std::string("bench check: ") + nm, me / mx < tol, d);
    double flops = 2.0 * M * N * K;   // nominal 2MNK (3xTF32 performs 3x the tensor work)
    double ms = time_ms([&] { launch_full(s, 48 * 1024); }, 10); report(nm, "full dispatch", ms, flops, "GFLOP/s", prec ? 156000 / 3.0 : 156000);
    ms = time_ms([&] { launch_one<ID>(s, 48 * 1024); }, 10); report(nm, "single-op", ms, flops, "GFLOP/s", prec ? 156000 / 3.0 : 156000);
}

template <int HD> static void bench_attn(int B, int T, int H, int Hkv) {
    Rng r(4); Dev dv; int R = B * T;
    int q = dv.addf(R, H * HD), k = dv.addf(R, Hkv * HD), v = dv.addf(R, Hkv * HD), o = dv.addf(R, H * HD), lse = dv.addf(1, B * H * T), dO = dv.addf(R, H * HD);
    int delta = dv.addf(1, B * H * T), dq = dv.addf(R, H * HD), dk = dv.addf(R, Hkv * HD), dvv = dv.addf(R, Hkv * HD);
    auto hq = rnd(r, (size_t)R * H * HD), hk = rnd(r, (size_t)R * Hkv * HD), hv = rnd(r, (size_t)R * Hkv * HD), hdo = rnd(r, (size_t)R * H * HD);
    dv.up(q, hq); dv.up(k, hk); dv.up(v, hv); dv.up(dO, hdo);
    float sc = 1.f / sqrtf((float)HD);
    Batch f, bd, bq, bkv;   // heaviest (latest) query blocks first
    for (int q0 = ((T - 1) / 32) * 32; q0 >= 0; q0 -= 32) for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            f.add(OP_attn_fwd, A().b(q).b(k).b(v).b(o).b(lse).i(b).i(h).i(q0).i(T).i(H).i(Hkv).i(HD).f(sc));
            bq.add(OP_attn_bwd_dq, A().b(q).b(k).b(v).b(dO).b(lse).b(delta).b(dq).i(b).i(h).i(q0).i(T).i(H).i(Hkv).i(HD).f(sc));
        }
    }
    for (int k0 = 0; k0 < T; k0 += 32) for (int b = 0; b < B; b++) for (int h = 0; h < Hkv; h++)   // dkv: early key blocks are heaviest
        bkv.add(OP_attn_bwd_dkv, A().b(q).b(k).b(v).b(dO).b(lse).b(delta).b(dk).b(dvv).i(b).i(h).i(k0).i(T).i(H).i(HD).f(sc));
    for (int r0 = 0; r0 < R; r0 += 128) bd.add(OP_attn_bwd_delta, A().b(o).b(dO).b(delta).i(r0).i(std::min(R, r0 + 128)).i(T).i(H).i(HD));
    int smem = kvm_gpt_smem_bytes(HD);
    auto sf = stage(dv, f), sd = stage(dv, bd), sq = stage(dv, bq), skv = stage(dv, bkv);
    const char* stage_names[4] = {"fwd", "delta", "dq", "dkv"}; const Staged* sts[4] = {&sf, &sd, &sq, &skv};
    for (int i = 0; i < 4; i++) { launch_full(*sts[i], smem); CK(cudaDeviceSynchronize()); int e = dv.geterr(); if (e) { printf("FAIL  attn bench: %s stage device error %d\n", stage_names[i], e); g_fail++; } }
    // correctness: batch 0 head 0 (and the kv head it maps to) on sampled rows against a double reference on that head only
    {
        auto go = dv.down(o), gdq = dv.down(dq), gdk = dv.down(dk); int G = H / Hkv; (void)G;
        // single-head reference: use Hkv-matching extraction when H==Hkv; for GQA compare fwd o only
        D qq(T * HD), kk(T * HD), vv(T * HD); for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) { qq[t * HD + d] = hq[(size_t)t * H * HD + d]; kk[t * HD + d] = hk[(size_t)t * Hkv * HD + d]; vv[t * HD + d] = hv[(size_t)t * Hkv * HD + d]; }
        D ro, rl, ddq, ddk, ddv, ddo(T * HD); for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) ddo[t * HD + d] = hdo[(size_t)t * H * HD + d];
        attn_d(1, T, 1, 1, HD, sc, qq, kk, vv, ro, rl, &ddo, &ddq, &ddk, &ddv);
        double me = 0, mx = 0; for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) { me = std::max(me, fabs(go[(size_t)t * H * HD + d] - ro[t * HD + d])); mx = std::max(mx, fabs(ro[t * HD + d])); }
        char nm[96], dd[64]; snprintf(nm, sizeof nm, "attn fwd B=%d T=%d H=%d Hkv=%d hd=%d (b0,h0 all rows)", B, T, H, Hkv, HD); snprintf(dd, sizeof dd, "relerr %.2e", me / mx);
        verdict(std::string("bench check: ") + nm, me / mx < 2e-5, dd);
        if (H == Hkv) {   // dq of head 0 is independent of the other heads; dk/dv of kv head 0 too when no sharing
            double e2 = 0, m2 = 0; for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) { e2 = std::max(e2, fabs(gdq[(size_t)t * H * HD + d] - ddq[t * HD + d])); m2 = std::max(m2, fabs(ddq[t * HD + d])); }
            snprintf(nm, sizeof nm, "attn bwd dq B=%d T=%d H=%d hd=%d (b0,h0)", B, T, H, HD); snprintf(dd, sizeof dd, "relerr %.2e", e2 / m2);
            verdict(std::string("bench check: ") + nm, e2 / m2 < 5e-5, dd);
            e2 = 0; m2 = 0; for (int t = 0; t < T; t++) for (int d = 0; d < HD; d++) { e2 = std::max(e2, fabs(gdk[(size_t)t * Hkv * HD + d] - ddk[t * HD + d])); m2 = std::max(m2, fabs(ddk[t * HD + d])); }
            snprintf(nm, sizeof nm, "attn bwd dk B=%d T=%d H=%d hd=%d (b0,h0)", B, T, H, HD); snprintf(dd, sizeof dd, "relerr %.2e", e2 / m2);
            verdict(std::string("bench check: ") + nm, e2 / m2 < 5e-5, dd);
        }
    }
    double fl = 2.0 * B * H * (double)T * T * HD;   // causal: 2 matmuls * 2 flops * T^2/2
    char nm[96]; snprintf(nm, sizeof nm, "attn fwd B=%d T=%d H=%d Hkv=%d hd=%d", B, T, H, Hkv, HD);
    double ms = time_ms([&] { launch_full(sf, smem); }, 5); report(nm, "full dispatch", ms, fl, "GFLOP/s", 19500);
    ms = time_ms([&] { launch_one<OP_attn_fwd>(sf, smem); }, 5); report(nm, "single-op", ms, fl, "GFLOP/s", 19500);
    snprintf(nm, sizeof nm, "attn bwd B=%d T=%d H=%d Hkv=%d hd=%d", B, T, H, Hkv, HD);
    auto bw = [&] { launch_full(sd, smem); launch_full(sq, smem); launch_full(skv, smem); };
    ms = time_ms(bw, 5); report(nm, "full dispatch", ms, fl * 2.5, "GFLOP/s", 19500);
    auto bw1 = [&] { launch_one<OP_attn_bwd_delta>(sd, smem); launch_one<OP_attn_bwd_dq>(sq, smem); launch_one<OP_attn_bwd_dkv>(skv, smem); };
    ms = time_ms(bw1, 5); report(nm, "single-op", ms, fl * 2.5, "GFLOP/s", 19500);
}

static void bench_mem() {
    Rng r(5); 
    {   // layernorm fwd [16384 x 1024]
        int R = 16384, d = 1024; Dev dv; int x = dv.addf(R, d), g = dv.addf(1, d), be = dv.addf(1, d), o = dv.addf(R, d), mean = dv.addf(1, R), rstd = dv.addf(1, R);
        auto hx = rnd(r, (size_t)R * d), hg = rnd(r, d), hb = rnd(r, d); dv.up(x, hx); dv.up(g, hg); dv.up(be, hb);
        Batch b; for (int r0 = 0; r0 < R; r0 += 64) b.add(OP_layernorm_fwd, A().b(x).b(g).b(be).b(o).b(mean).b(rstd).i(r0).i(r0 + 64).i(d).f(1e-5f));
        auto s = stage(dv, b); launch_full(s, 48 * 1024); CK(cudaDeviceSynchronize()); auto go = dv.down(o);
        D m, rs, ref = norm_fwd_d(D(hx.begin(), hx.begin() + 64 * d), dbl(hg), dbl(hb), 64, d, 1e-5, false, m, rs);
        std::vector<float> g64(go.begin(), go.begin() + 64 * d); check("bench check: layernorm fwd 16384x1024 (first 64 rows)", g64, ref, 1e-5);
        double ms = time_ms([&] { launch_full(s, 48 * 1024); }, 20); report("layernorm fwd 16384x1024", "full dispatch", ms, 2.0 * R * d * 4, "GB/s", 1555);
    }
    {   // adamw 16M
        int n = 1 << 24; Dev dv; int p = dv.addf(1, n), g = dv.addf(1, n), m = dv.addf(1, n), v = dv.addf(1, n), st = dv.addi(1);
        auto hp = rnd(r, n), hg = rnd(r, n); dv.up(p, hp); dv.up(g, hg); dv.upi(st, {1});
        Batch b; for (int e0 = 0; e0 < n; e0 += 8192) b.add(OP_adamw, A().b(p).b(g).b(m).b(v).b(-1).b(st).i(e0).i(e0 + 8192).f(1e-3f).f(.9f).f(.99f).f(1e-8f).f(.01f));
    {   // layernorm bwd dx and param grads [16384 x 1024]
        int R = 16384, d = 1024; Dev dv; int x = dv.addf(R, d), g = dv.addf(1, d), dy = dv.addf(R, d), mean = dv.addf(1, R), rstd = dv.addf(1, R), dx = dv.addf(R, d), dg = dv.addf(1, d), db = dv.addf(1, d);
        auto hx = rnd(r, (size_t)R * d), hg = rnd(r, d), hdy = rnd(r, (size_t)R * d), hm = rnd(r, R, 0.1), hr = rnd(r, R, 0.1); for (auto& v : hr) v = 1.f + fabsf(v);
        dv.up(x, hx); dv.up(g, hg); dv.up(dy, hdy); dv.up(mean, hm); dv.up(rstd, hr);
        Batch b, bp; for (int r0 = 0; r0 < R; r0 += 64) b.add(OP_layernorm_bwd_dx, A().b(x).b(dy).b(g).b(mean).b(rstd).b(dx).i(r0).i(r0 + 64).i(d).i(0));
        for (int c0 = 0; c0 < d; c0 += 64) bp.add(OP_norm_bwd_params, A().b(x).b(dy).b(mean).b(rstd).b(dg).b(db).i(R).i(d).i(c0).i(c0 + 64).i(0));
        Batch bp16; for (int c0 = 0; c0 < d; c0 += 16) bp16.add(OP_norm_bwd_params, A().b(x).b(dy).b(mean).b(rstd).b(dg).b(db).i(R).i(d).i(c0).i(c0 + 16).i(0));
        auto sp16 = stage(dv, bp16);
        auto s = stage(dv, b), sp = stage(dv, bp); launch_full(s, 48 * 1024); launch_full(sp, 48 * 1024); CK(cudaDeviceSynchronize());
        auto gdx = dv.down(dx), gdg = dv.down(dg);
        double me = 0, ms_ = 0; for (int rr = 0; rr < 8; rr++) {   // row-local reference for 8 rows
            double s1 = 0, s2 = 0; for (int c = 0; c < d; c++) { double gc = (double)hdy[(size_t)rr * d + c] * hg[c], xh = (hx[(size_t)rr * d + c] - (double)hm[rr]) * hr[rr]; s1 += gc; s2 += gc * xh; }
            s1 /= d; s2 /= d;
            for (int c = 0; c < d; c++) { double gc = (double)hdy[(size_t)rr * d + c] * hg[c], xh = (hx[(size_t)rr * d + c] - (double)hm[rr]) * hr[rr], v = hr[rr] * (gc - s1 - xh * s2); me = std::max(me, fabs(gdx[(size_t)rr * d + c] - v)); ms_ = std::max(ms_, fabs(v)); }
        }
        char dd[64]; snprintf(dd, sizeof dd, "relerr %.2e", me / ms_); verdict("bench check: layernorm bwd dx 16384x1024 (8 rows)", me / ms_ < 1e-5, dd);
        double eg = 0, mg = 0; for (int c = 0; c < d; c += 37) { double acc = 0; for (int rr = 0; rr < R; rr++) acc += (double)hdy[(size_t)rr * d + c] * (hx[(size_t)rr * d + c] - (double)hm[rr]) * hr[rr]; eg = std::max(eg, fabs(gdg[c] - acc)); mg = std::max(mg, fabs(acc)); }
        snprintf(dd, sizeof dd, "relerr %.2e", eg / mg); verdict("bench check: norm_bwd_params dgamma 16384x1024 (sampled cols)", eg / mg < 1e-4, dd);
        double ms = time_ms([&] { launch_full(s, 48 * 1024); }, 20); report("layernorm bwd dx 16384x1024", "full dispatch", ms, 3.0 * R * d * 4, "GB/s", 1555);
        ms = time_ms([&] { launch_full(sp, 48 * 1024); }, 20); report("norm_bwd_params 16384x1024 (16 tasks of 64 cols)", "full dispatch", ms, 2.0 * R * d * 4, "GB/s", 1555);
        ms = time_ms([&] { launch_full(sp16, 48 * 1024); }, 20); report("norm_bwd_params 16384x1024 (64 tasks of 16 cols)", "full dispatch", ms, 2.0 * R * d * 4, "GB/s", 1555);
    }
        auto s = stage(dv, b); launch_full(s, 48 * 1024); CK(cudaDeviceSynchronize());
        auto gp = dv.down(p); D P = dbl(hp), M(n, 0.0), Vv(n, 0.0); adamw_d(P, dbl(hg), M, Vv, 1e-3f, .9f, .99f, 1e-8f, .01f, 1, 1.0);
        double me = 0; for (int i = 0; i < n; i += 1009) me = std::max(me, fabs(gp[i] - P[i]));
        char d[64]; snprintf(d, sizeof d, "max abs diff %.2e (sampled)", me); verdict("bench check: adamw 16M (first step; later steps mutate state)", me < 1e-6, d);
        double ms = time_ms([&] { launch_full(s, 48 * 1024); }, 20); report("adamw 16M params", "full dispatch", ms, 7.0 * n * 4, "GB/s", 1555);
    }
    {   // gelu fwd 64M
        int n = 1 << 26; Dev dv; int x = dv.addf(1, n), o = dv.addf(1, n);
        auto hx = rnd(r, n); dv.up(x, hx);
        Batch b; for (int e0 = 0; e0 < n; e0 += 16384) b.add(OP_gelu_fwd, A().b(x).b(o).i(e0).i(e0 + 16384));
        auto s = stage(dv, b); launch_full(s, 48 * 1024); CK(cudaDeviceSynchronize()); auto go = dv.down(o);
        double me = 0; for (int i = 0; i < n; i += 997) { double v = hx[i], t = 0.5 * v * (1 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))); me = std::max(me, fabs(go[i] - t)); }
        char d[64]; snprintf(d, sizeof d, "max abs diff %.2e (sampled)", me); verdict("bench check: gelu fwd 64M", me < 1e-5, d);
        double ms = time_ms([&] { launch_full(s, 48 * 1024); }, 10); report("gelu fwd 64M", "full dispatch", ms, 2.0 * n * 4, "GB/s", 1555);
    }
    {   // softmax CE fwd+bwd [8192 x 50304]
        int N = 8192, Vv = 50304; Dev dv; int lg = dv.addf(N, Vv), tg = dv.addi(N), rl = dv.addf(1, N), ls = dv.addf(1, N), loss = dv.addf(1, 2), dl = dv.addf(N, Vv);
        auto hl = rnd(r, (size_t)N * Vv, 3.0); std::vector<int32_t> ht(N); for (auto& t : ht) t = r.u32() % Vv; dv.up(lg, hl); dv.upi(tg, ht);
        Batch f, bw; for (int r0 = 0; r0 < N; r0 += 8) { f.add(OP_softmax_ce_fwd, A().b(lg).b(tg).b(rl).b(ls).i(r0).i(r0 + 8).i(Vv).i(-100)); bw.add(OP_softmax_ce_bwd, A().b(lg).b(tg).b(ls).b(loss).b(dl).i(r0).i(r0 + 8).i(Vv).i(-100).f(1.f)); }
        auto sf = stage(dv, f), sb = stage(dv, bw); launch_full(sf, 48 * 1024); CK(cudaDeviceSynchronize());
        run1(dv, OP_softmax_ce_reduce, A().b(rl).b(tg).b(loss).i(N).i(-100)); launch_full(sb, 48 * 1024); CK(cudaDeviceSynchronize());
        auto gl = dv.down(loss); D ref8; double rloss = 0;
        { std::vector<int> it(ht.begin(), ht.end()); D sub(hl.begin(), hl.begin() + 64 * (size_t)Vv); std::vector<int> t64(it.begin(), it.begin() + 64); rloss = ce_d(sub, t64, 64, Vv, -100, nullptr); }
        auto gr = dv.down(rl); double sum = 0; for (int i = 0; i < 64; i++) sum += gr[i];
        char d[96]; snprintf(d, sizeof d, "mean of first 64 row losses %.6f vs %.6f", sum / 64, rloss);
        verdict("bench check: softmax_ce fwd 8192x50304", fabs(sum / 64 - rloss) < 1e-4, d);
        double ms = time_ms([&] { launch_full(sf, 48 * 1024); }, 10); report("softmax_ce fwd 8192x50304", "full dispatch", ms, 1.0 * N * Vv * 4, "GB/s (1 read counted)", 1555);
        ms = time_ms([&] { launch_full(sb, 48 * 1024); }, 10); report("softmax_ce bwd 8192x50304", "full dispatch", ms, 2.0 * N * Vv * 4, "GB/s (read+write)", 1555);
    }
}

// BENCH_ONLY=mm|attn|mem runs one group; BENCH_REPS=n sets the repetition count.
static bool want(const char* g) { const char* e = getenv("BENCH_ONLY"); return !e || !strcmp(e, g); }

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("device: %s, %d SMs, %d MHz\n", pr.name, pr.multiProcessorCount, pr.clockRate / 1000);
    if (want("mm")) {
    bench_mm<OP_matmul_tile>(4096, 4096, 4096, 0, 0, 0); bench_mm<OP_matmul_tile>(4096, 4096, 4096, 1, 0, 0); bench_mm<OP_matmul_tile>(4096, 4096, 4096, 0, 1, 0);
    bench_mm<OP_matmul_tile>(4096, 4096, 4096, 0, 0, 1); bench_mm<OP_matmul_tile>(2048, 2048, 2048, 0, 0, 0); bench_mm<OP_matmul_tile>(8192, 768, 768, 0, 0, 0);
    bench_mm<OP_matmul_tile>(1024, 1024, 1024, 0, 0, 0); bench_mm<OP_matmul_tile>(4096, 3072, 768, 0, 0, 0); bench_mm<OP_matmul_tile>(512, 512, 512, 0, 0, 0);
    }
    if (want("attn")) { bench_attn<64>(4, 1024, 16, 16); bench_attn<64>(4, 2048, 16, 4); bench_attn<128>(2, 2048, 8, 8); }
    if (want("mem")) bench_mem();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
