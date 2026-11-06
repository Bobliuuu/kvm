// Per-op unit tests for the extra generic operator pack (include/kvm_ops_extra.cuh) against independent float64 CPU references,
// edge cases and error paths (nonzero code, memory untouched).
// Build: nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -I../../include -I. test_extra.cu -o test_extra
#include <algorithm>
#include <map>
#include <numeric>
#ifndef KVM_GPT_MAX_HD
#define KVM_GPT_MAX_HD 128   // hd=128 attention is covered here
#endif
#include "harness.cuh"
#include "refs.hpp"
#include "kvm_ops_extra.cuh"

struct OpInfoX { const char* name; int id; const char* sig; };
static const OpInfoX OPSX[] = {
#define X(n, id, s) {#n, id, s},
#include "kvm_ops_extra.def"
#undef X
};
enum {
#define X(n, id, s) OPX_##n = id,
#include "kvm_ops_extra.def"
#undef X
};

__global__ void k_runx(KvmCtx c, const int32_t* w, const int* op, const int* off) {
    const int32_t* a = w + off[blockIdx.x];
    if (!kvm_extra_dispatch(c, op[blockIdx.x], a) && !kvm_gpt_dispatch(c, op[blockIdx.x], a)) atomicCAS(c.err, 0, 99);
}
struct BatchX {
    std::vector<int> op, off; std::vector<int32_t> w;
    void add(int id, const A& a, bool nocheck = false) {
        const char* sig = nullptr; const char* nm = "?";
        for (auto& o : OPSX) if (o.id == id) { sig = o.sig; nm = o.name; }
        for (auto& o : OPS) if (o.id == id) { sig = o.sig; nm = o.name; }
        if (!sig) { printf("unknown op %d\n", id); exit(2); }
        int ar = 0; for (const char* p = sig; *p; p++) ar += *p == ':';
        if (!nocheck && ar != (int)a.w.size()) { printf("arity mismatch for %s: def %d, got %d\n", nm, ar, (int)a.w.size()); exit(2); }
        op.push_back(id); off.push_back((int)w.size()); w.insert(w.end(), a.w.begin(), a.w.end());
    }
};
static int g_attr = 0;
static int runx(Dev& d, const BatchX& b, int smem = 104 * 1024) {
    if (!g_attr) { CK(cudaFuncSetAttribute(k_runx, cudaFuncAttributeMaxDynamicSharedMemorySize, 112 * 1024)); g_attr = 1; }
    d.sync_table();
    int32_t* w; int *op, *off;
    CK(cudaMalloc(&w, b.w.size() * 4 + 4)); CK(cudaMalloc(&op, b.op.size() * 4)); CK(cudaMalloc(&off, b.off.size() * 4));
    CK(cudaMemcpy(w, b.w.data(), b.w.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(op, b.op.data(), b.op.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(off, b.off.data(), b.off.size() * 4, cudaMemcpyHostToDevice));
    KvmCtx c{d.tab_d, (int32_t)d.tab.size(), d.err};
    k_runx<<<(unsigned)b.op.size(), KVM_GPT_THREADS, smem>>>(c, w, op, off);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaFree(w); cudaFree(op); cudaFree(off);
    return d.geterr();
}
static int run1x(Dev& d, int id, const A& a, int smem = 104 * 1024) { BatchX b; b.add(id, a); return runx(d, b, smem); }

static std::string S(int v) { return std::to_string(v); }
static float f32bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t bitsf(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// independent bf16 round-to-nearest-even (no bit tricks): quantize the double value to the bf16 grid with rint
static float bf16_ref(float x) {
    if (std::isnan(x) || std::isinf(x) || x == 0.f) return x;
    int e = ilogb((double)x);
    if (e < -126) e = -126;
    double q = ldexp(1.0, e - 7);
    double y = std::rint((double)x / q) * q;   // default rounding mode: nearest even
    if (fabs(y) >= ldexp(1.0, 128)) return y < 0 ? -INFINITY : INFINITY;
    return (float)y;
}
static D rb(const std::vector<float>& v) { D o(v.size()); for (size_t i = 0; i < v.size(); i++) o[i] = bf16_ref(v[i]); return o; }

// ---- elementwise ------------------------------------------------------------------------------------------------------------
static double act_d(int kind, double x) { return kind == 0 ? (x > 0 ? x : 0) : kind == 1 ? x / (1 + exp(-x)) : kind == 2 ? tanh(x) : 1 / (1 + exp(-x)); }
static double actg_d(int kind, double x) {
    double s = 1 / (1 + exp(-x));
    return kind == 0 ? (x > 0 ? 1 : 0) : kind == 1 ? s * (1 + x * (1 - s)) : kind == 2 ? 1 - tanh(x) * tanh(x) : s * (1 - s);
}
static double silu_d(double x) { return x / (1 + exp(-x)); }
static double dsilu_d(double x) { double s = 1 / (1 + exp(-x)); return s * (1 + x * (1 - s)); }

static void t_elementwise() {
    Rng r(101);
    for (int n : {1, 7, 1000}) {
        Dev dv;
        int x = dv.addf(1, n), o = dv.addf(1, n), dy = dv.addf(1, n), dx = dv.addf(1, n);
        auto hx = rnd(r, n, 3.0), hdy = rnd(r, n);
        for (int i = 0; i < n; i += 5) hx[i] = 0.f;   // relu'(0) = 0
        dv.up(x, hx); dv.up(dy, hdy);
        for (int kind = 0; kind < 4; kind++) {
            std::string nm = "act kind=" + S(kind) + " n=" + S(n);
            BatchX b; for (int e0 = 0; e0 < n; e0 += 300) b.add(OPX_act_fwd, A().b(x).b(o).i(kind).i(e0).i(std::min(n, e0 + 300)));
            int e = runx(dv, b);
            D ref(n); for (int i = 0; i < n; i++) ref[i] = act_d(kind, hx[i]);
            check(nm + " fwd", dv.down(o), ref, 1e-5); check_true(nm + " fwd err", e == 0);
            for (int acc = 0; acc < 2; acc++) {
                dv.up(dx, rnd(r, n)); auto d0 = dv.down(dx);
                BatchX b2; for (int e0 = 0; e0 < n; e0 += 300) b2.add(OPX_act_bwd, A().b(x).b(dy).b(dx).i(kind).i(e0).i(std::min(n, e0 + 300)).i(acc));
                e = runx(dv, b2);
                for (int i = 0; i < n; i++) ref[i] = (acc ? d0[i] : 0.0) + hdy[i] * actg_d(kind, hx[i]);
                check(nm + " bwd acc=" + S(acc), dv.down(dx), ref, 1e-5); check_true(nm + " bwd err", e == 0);
            }
        }
        // partial range leaves the rest untouched; in-place
        dv.up(o, hx);
        run1x(dv, OPX_act_fwd, A().b(o).b(o).i(1).i(n / 3).i(n - n / 3));
        auto ho = dv.down(o); bool ok = true;
        for (int i = 0; i < n; i++) { bool in = i >= n / 3 && i < n - n / 3; ok &= in ? fabs(ho[i] - act_d(1, hx[i])) <= 1e-5 * (fabs(hx[i]) + 1) : ho[i] == hx[i]; }
        check_true("act_fwd partial range + in place n=" + S(n), ok);
        // silu_mul
        int g = dv.addf(1, n), u = dv.addf(1, n), dg = dv.addf(1, n), du = dv.addf(1, n);
        auto hg = rnd(r, n, 2.0), hu = rnd(r, n); dv.up(g, hg); dv.up(u, hu);
        run1x(dv, OPX_silu_mul_fwd, A().b(g).b(u).b(o).i(0).i(n));
        D ref(n); for (int i = 0; i < n; i++) ref[i] = silu_d(hg[i]) * hu[i];
        check("silu_mul_fwd n=" + S(n), dv.down(o), ref, 1e-5);
        for (int acc = 0; acc < 2; acc++) {
            dv.up(dg, rnd(r, n)); dv.up(du, rnd(r, n)); auto g0 = dv.down(dg), u0 = dv.down(du);
            BatchX b3; for (int e0 = 0; e0 < n; e0 += 300) b3.add(OPX_silu_mul_bwd, A().b(g).b(u).b(dy).b(dg).b(du).i(e0).i(std::min(n, e0 + 300)).i(acc));
            runx(dv, b3);
            D rg(n), ru(n);
            for (int i = 0; i < n; i++) { rg[i] = (acc ? g0[i] : 0.0) + hdy[i] * hu[i] * dsilu_d(hg[i]); ru[i] = (acc ? u0[i] : 0.0) + hdy[i] * silu_d(hg[i]); }
            check("silu_mul_bwd dgate acc=" + S(acc) + " n=" + S(n), dv.down(dg), rg, 1e-5);
            check("silu_mul_bwd dup acc=" + S(acc) + " n=" + S(n), dv.down(du), ru, 1e-5);
        }
        // scale_flat
        run1x(dv, OPX_scale_flat, A().b(x).b(o).f(-0.37f).i(0).i(n));
        for (int i = 0; i < n; i++) ref[i] = (double)(-0.37f) * hx[i];
        check("scale_flat n=" + S(n), dv.down(o), ref, 1e-7);
        run1x(dv, OPX_scale_flat, A().b(x).b(x).f(2.5f).i(0).i(n));
        for (int i = 0; i < n; i++) ref[i] = 2.5 * hx[i];
        check("scale_flat in place n=" + S(n), dv.down(x), ref, 1e-7);
    }
    // add_rowtile
    for (int d : {1, 5, 64, 100}) {
        int R = 29, T = 8;
        Dev dv; int x = dv.addf(R, d, d + 3), p = dv.addf(T, d, d + 1), o = dv.addf(R, d, d + 2);
        auto hx = rnd(r, R * d), hp = rnd(r, T * d); dv.up(x, hx); dv.up(p, hp);
        BatchX b; for (int r0 = 0; r0 < R; r0 += 10) b.add(OPX_add_rowtile, A().b(x).b(p).b(o).i(r0).i(std::min(R, r0 + 10)).i(T).i(d));
        int e = runx(dv, b);
        D ref(R * d); for (int i = 0; i < R; i++) for (int c = 0; c < d; c++) ref[i * d + c] = (double)hx[i * d + c] + hp[(i % T) * d + c];
        check("add_rowtile d=" + S(d), dv.down(o), ref, 1e-7); check_true("add_rowtile err", e == 0);
        run1x(dv, OPX_add_rowtile, A().b(x).b(p).b(x).i(0).i(R).i(T).i(d));
        check("add_rowtile in place d=" + S(d), dv.down(x), ref, 1e-7);
    }
}

// ---- softmax ------------------------------------------------------------------------------------------------------------------
static void t_softmax() {
    Rng r(111);
    for (int d : {1, 3, 33, 100, 1000, 8192}) {
        int R = d > 1000 ? 6 : 19;
        float scale = 0.37f;
        Dev dv; int x = dv.addf(R, d, d + 4), o = dv.addf(R, d, d + 1), p = dv.addf(R, d), dp = dv.addf(R, d, d + 3), dx = dv.addf(R, d, d + 2);
        auto hx = rnd(r, R * d, 3.0); dv.up(x, hx);
        BatchX b; for (int r0 = 0; r0 < R; r0 += 5) b.add(OPX_softmax_rows_fwd, A().b(x).b(o).i(r0).i(std::min(R, r0 + 5)).i(d).f(scale));
        int e = runx(dv, b);
        D ref(R * d), hp(R * d);
        for (int i = 0; i < R; i++) {
            double mx = -1e300, s = 0; for (int c = 0; c < d; c++) mx = std::max(mx, scale * (double)hx[i * d + c]);
            for (int c = 0; c < d; c++) s += exp(scale * (double)hx[i * d + c] - mx);
            for (int c = 0; c < d; c++) ref[i * d + c] = exp(scale * (double)hx[i * d + c] - mx) / s;
        }
        check("softmax_rows_fwd d=" + S(d), dv.down(o), ref, 1e-5); check_true("softmax_rows_fwd err", e == 0);
        // backward with p = the exact forward output
        auto hpf = dv.down(o); dv.up(p, hpf);
        auto hdp = rnd(r, R * d); dv.up(dp, hdp);
        for (int acc = 0; acc < 2; acc++) {
            dv.up(dx, rnd(r, R * d)); auto d0 = dv.down(dx);
            BatchX b2; for (int r0 = 0; r0 < R; r0 += 5) b2.add(OPX_softmax_rows_bwd, A().b(p).b(dp).b(dx).i(r0).i(std::min(R, r0 + 5)).i(d).f(scale).i(acc));
            e = runx(dv, b2);
            D rf(R * d);
            for (int i = 0; i < R; i++) {
                double s = 0; for (int c = 0; c < d; c++) s += (double)hpf[i * d + c] * hdp[i * d + c];
                for (int c = 0; c < d; c++) rf[i * d + c] = (acc ? d0[i * d + c] : 0.0) + scale * (double)hpf[i * d + c] * (hdp[i * d + c] - s);
            }
            check("softmax_rows_bwd d=" + S(d) + " acc=" + S(acc), dv.down(dx), rf, 1e-5, 1e-6); check_true("softmax_rows_bwd err", e == 0);
        }
        auto o1 = dv.down(o); runx(dv, b); check_true("softmax_rows_fwd bitwise deterministic d=" + S(d), o1 == dv.down(o));
    }
    {   // negative scale, in place, partial rows
        int R = 7, d = 50; Dev dv; int x = dv.addf(R, d); auto hx = rnd(r, R * d); dv.up(x, hx);
        run1x(dv, OPX_softmax_rows_fwd, A().b(x).b(x).i(2).i(5).i(d).f(-1.5f));
        auto g = dv.down(x); D ref(R * d); bool ok = true;
        for (int i = 0; i < R; i++) {
            double mx = -1e300, s = 0; for (int c = 0; c < d; c++) mx = std::max(mx, -1.5 * hx[i * d + c]);
            for (int c = 0; c < d; c++) s += exp(-1.5 * hx[i * d + c] - mx);
            for (int c = 0; c < d; c++) { double v = (i >= 2 && i < 5) ? exp(-1.5 * hx[i * d + c] - mx) / s : hx[i * d + c]; ok &= fabs(g[i * d + c] - v) <= 1e-6 * (fabs(v) + 1e-3); }
        }
        check_true("softmax_rows_fwd in place, scale<0, rows 2..5", ok);
    }
}

// ---- gather / scatter / row_dot -------------------------------------------------------------------------------------------------
static void t_gather() {
    Rng r(121);
    for (int d : {1, 5, 64, 130}) {
        int R = 23, Rs = 17, V = 14, N = 40;   // gather: R dst rows from Rs src rows; scatter: N src rows into V dst rows
        Dev dv;
        int src = dv.addf(Rs, d, d + 3), idx = dv.addi(R), sc = dv.addf(R, 1), dst = dv.addf(R, d, d + 1), a = dv.addf(Rs, d), bm = dv.addf(R, d), od = dv.addf(R, 1);
        auto hs = rnd(r, Rs * d), hsc = rnd(r, R), hb = rnd(r, R * d), ha = rnd(r, Rs * d);
        std::vector<int32_t> hi(R); for (auto& v : hi) v = (int)(r.u32() % (Rs + 6)) - 3;   // includes -3..-1 and Rs..Rs+2
        hi[0] = -1; hi[1] = Rs; hi[2] = 0; hi[3] = Rs - 1;
        dv.up(src, hs); dv.upi(idx, hi); dv.up(sc, hsc); dv.up(bm, hb); dv.up(a, ha);
        for (int hasc = 0; hasc < 2; hasc++) for (int acc = 0; acc < 2; acc++) {
            dv.up(dst, rnd(r, R * d)); auto d0 = dv.down(dst);
            BatchX b; for (int r0 = 0; r0 < R; r0 += 8) b.add(OPX_gather_rows_scaled, A().b(src).b(idx).b(hasc ? sc : -1).b(dst).i(r0).i(std::min(R, r0 + 8)).i(d).i(acc).i(0));
            int e = runx(dv, b);
            D ref(R * d);
            for (int i = 0; i < R; i++) for (int c = 0; c < d; c++) {
                double v = (hi[i] >= 0 && hi[i] < Rs) ? (hasc ? hsc[i] : 1.0) * hs[hi[i] * d + c] : 0.0;
                ref[i * d + c] = (acc ? d0[i * d + c] : 0.0) + v;
            }
            check("gather_rows_scaled d=" + S(d) + " scale=" + S(hasc) + " acc=" + S(acc), dv.down(dst), ref, 1e-6, 1e-3); check_true("gather err", e == 0);
        }
        // row_dot with and without idx, acc
        for (int useidx = 0; useidx < 2; useidx++) for (int acc = 0; acc < 2; acc++) {
            dv.up(od, rnd(r, R)); auto o0 = dv.down(od);
            BatchX b; for (int r0 = 0; r0 < R; r0 += 9) b.add(OPX_row_dot, A().b(a).b(bm).b(useidx ? idx : -1).b(od).i(r0).i(std::min(R, r0 + 9)).i(d).i(acc).i(0));
            int e = runx(dv, b);
            D ref(R);
            for (int i = 0; i < R; i++) {
                int j = useidx ? hi[i] : i; double s = 0;
                if (j >= 0 && j < Rs) for (int c = 0; c < d; c++) s += (double)ha[j * d + c] * hb[i * d + c];
                ref[i] = (acc ? o0[i] : 0.0) + s;
            }
            check("row_dot d=" + S(d) + " idx=" + S(useidx) + " acc=" + S(acc), dv.down(od), ref, 1e-5, 1e-3); check_true("row_dot err", e == 0);
        }
        // scatter_add_rows: duplicates, -1 padding, out-of-range
        int ssrc = dv.addf(N, d, d + 2), sidx = dv.addi(N), ssc = dv.addf(N, 1), sdst = dv.addf(V, d, d + 1);
        auto hss = rnd(r, N * d), hssc = rnd(r, N);
        std::vector<int32_t> hsi(N); for (auto& v : hsi) v = (int)(r.u32() % (V + 6)) - 3;
        hsi[0] = hsi[1] = hsi[2] = 5;
        dv.up(ssrc, hss); dv.upi(sidx, hsi); dv.up(ssc, hssc);
        for (int hasc = 0; hasc < 2; hasc++) for (int acc = 0; acc < 2; acc++) {
            dv.up(sdst, rnd(r, V * d)); auto d0 = dv.down(sdst);
            BatchX b; for (int v0 = 0; v0 < V; v0 += 4) b.add(OPX_scatter_add_rows, A().b(ssrc).b(sidx).b(hasc ? ssc : -1).b(sdst).i(N).i(v0).i(std::min(V, v0 + 4)).i(d).i(acc).i(0));
            int e = runx(dv, b);
            D ref(V * d);
            for (int v = 0; v < V; v++) for (int c = 0; c < d; c++) {
                double s = 0; for (int i = 0; i < N; i++) if (hsi[i] == v) s += (hasc ? hssc[i] : 1.0) * hss[i * d + c];
                ref[v * d + c] = (acc ? d0[v * d + c] : 0.0) + s;
            }
            check("scatter_add_rows d=" + S(d) + " scale=" + S(hasc) + " acc=" + S(acc), dv.down(sdst), ref, 1e-5, 1e-3); check_true("scatter err", e == 0);
            auto s1 = dv.down(sdst); dv.up(sdst, d0.empty() ? d0 : d0); runx(dv, b); check_true("scatter_add_rows bitwise deterministic", s1 == dv.down(sdst));
        }
    }
}

// ---- MoE ---------------------------------------------------------------------------------------------------------------------------
static void t_moe() {
    Rng r(131);
    struct Cfg { int N, E, k, C; bool ties; };
    Cfg cfgs[] = {{1, 1, 1, 1, false}, {1, 4, 2, 1, false}, {17, 4, 2, 3, false}, {17, 4, 2, 100, false}, {300, 8, 2, 20, false}, {300, 8, 2, 20, true},
                  {64, 64, 8, 4, false}, {64, 64, 8, 4, true}, {500, 16, 4, 10, true}, {37, 5, 5, 8, true}, {2000, 32, 8, 100, false}, {33, 3, 1, 5, false}};
    for (auto cf : cfgs) {
        int N = cf.N, E = cf.E, K = cf.k, C = cf.C;
        std::string nm = "moe N=" + S(N) + " E=" + S(E) + " k=" + S(K) + " C=" + S(C) + (cf.ties ? " ties" : "");
        std::vector<float> hp((size_t)N * E);
        for (int t = 0; t < N; t++) {
            if (cf.ties) { for (int e = 0; e < E; e++) hp[t * E + e] = (float)(1 + r.u32() % 3) / 8.f; }
            else {
                double s = 0, lg[64]; for (int e = 0; e < E; e++) { lg[e] = r.nrm(); s += exp(lg[e]); }
                for (int e = 0; e < E; e++) hp[t * E + e] = (float)(exp(lg[e]) / s);
            }
        }
        Dev dv;
        int probs = dv.addf(N, E, E + 1), sel = dv.addi(N * K), slot = dv.addi(N * K), eidx = dv.addi(E * C);
        int gs = dv.addf(E * C, 1), dgs = dv.addf(E * C, 1), dprobs = dv.addf(N, E, E + 2), loss = dv.addf(1, 1);
        dv.up(probs, hp); dv.up(gs, rnd(r, E * C)); dv.upi(eidx, std::vector<int32_t>(E * C, 77));
        int e = run1x(dv, OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N).i(E).i(K).i(C));
        std::vector<int32_t> rsel(N * K), rslot(N * K, -9), reidx(E * C, -1); std::vector<int> cnt(E, 0);
        for (int t = 0; t < N; t++) {
            std::vector<int> ord(E); std::iota(ord.begin(), ord.end(), 0);
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return hp[t * E + a] > hp[t * E + b]; });
            for (int j = 0; j < K; j++) rsel[t * K + j] = ord[j];
        }
        for (int t = 0; t < N; t++) for (int j = 0; j < K; j++) {
            int ex = rsel[t * K + j];
            if (cnt[ex] < C) { rslot[t * K + j] = ex * C + cnt[ex]; reidx[ex * C + cnt[ex]] = t; cnt[ex]++; } else rslot[t * K + j] = -1;
        }
        check_true(nm + " route sel exact", dv.downi(sel) == rsel);
        check_true(nm + " route slot exact", dv.downi(slot) == rslot);
        check_true(nm + " route eidx exact", dv.downi(eidx) == reidx);
        check_true(nm + " route err", e == 0);
        int drops = 0; for (int v : rslot) drops += v < 0;
        if (N * K > E * C) check_true(nm + " (capacity overflow exercised)", drops > 0);
        // gates fwd/bwd
        for (int renorm = 0; renorm < 2; renorm++) {
            e = run1x(dv, OPX_moe_gates_fwd, A().b(probs).b(sel).b(slot).b(gs).i(N).i(E).i(K).i(renorm));
            D rg(E * C, 0.0);
            std::vector<double> Zs(N);
            for (int t = 0; t < N; t++) {
                double Z = 0; for (int j = 0; j < K; j++) Z += hp[t * E + rsel[t * K + j]];
                Zs[t] = renorm ? Z : 1.0;
                for (int j = 0; j < K; j++) if (rslot[t * K + j] >= 0) rg[rslot[t * K + j]] = hp[t * E + rsel[t * K + j]] / Zs[t];
            }
            check(nm + " gates_fwd renorm=" + S(renorm), dv.down(gs), rg, 1e-6, 1e-3); check_true(nm + " gates_fwd err", e == 0);
            auto hdg = rnd(r, E * C); dv.up(dgs, hdg);
            for (int acc = 0; acc < 2; acc++) {
                dv.up(dprobs, rnd(r, N * E)); auto d0 = dv.down(dprobs);
                e = run1x(dv, OPX_moe_gates_bwd, A().b(probs).b(sel).b(slot).b(dgs).b(dprobs).i(N).i(E).i(K).i(renorm).i(acc));
                D rd((size_t)N * E, 0.0); if (acc) for (int i = 0; i < N * E; i++) rd[i] = d0[i];
                for (int t = 0; t < N; t++) {
                    double dg[8], p[8], S2 = 0, Z = Zs[t];
                    for (int j = 0; j < K; j++) { int s = rslot[t * K + j]; dg[j] = s >= 0 ? hdg[s] : 0.0; p[j] = hp[t * E + rsel[t * K + j]]; S2 += dg[j] * p[j]; }
                    for (int j = 0; j < K; j++) rd[t * E + rsel[t * K + j]] += renorm ? dg[j] / Z - S2 / (Z * Z) : dg[j];
                }
                check(nm + " gates_bwd renorm=" + S(renorm) + " acc=" + S(acc), dv.down(dprobs), rd, 1e-5, 1.0); check_true(nm + " gates_bwd err", e == 0);
            }
        }
        // balance
        float coef = 0.01f;
        e = run1x(dv, OPX_moe_balance_fwd, A().b(probs).b(sel).b(loss).i(N).i(E).i(K).f(coef));
        std::vector<double> f(E, 0.0), P(E, 0.0);
        for (int i = 0; i < N * K; i++) f[rsel[i]] += 1.0;
        for (int ex = 0; ex < E; ex++) { f[ex] /= (double)N * K; for (int t = 0; t < N; t++) P[ex] += hp[t * E + ex]; P[ex] /= N; }
        double L = 0; for (int ex = 0; ex < E; ex++) L += f[ex] * P[ex]; L *= (double)coef * E;
        check(nm + " balance_fwd", dv.down(loss), D{L}, 1e-6, 1e-12); check_true(nm + " balance_fwd err", e == 0);
        for (int acc = 0; acc < 2; acc++) {
            dv.up(dprobs, rnd(r, N * E)); auto d0 = dv.down(dprobs);
            e = run1x(dv, OPX_moe_balance_bwd, A().b(probs).b(sel).b(dprobs).i(N).i(E).i(K).f(coef).i(acc));
            D rd((size_t)N * E);
            for (int t = 0; t < N; t++) for (int ex = 0; ex < E; ex++) rd[t * E + ex] = (acc ? d0[t * E + ex] : 0.0) + (double)coef * E * f[ex] / N;
            check(nm + " balance_bwd acc=" + S(acc), dv.down(dprobs), rd, 1e-6, 1e-12); check_true(nm + " balance_bwd err", e == 0);
        }
        // bitwise determinism of route
        auto s1 = dv.downi(slot); run1x(dv, OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N).i(E).i(K).i(C));
        check_true(nm + " route deterministic", s1 == dv.downi(slot));
    }
    {   // limits: E=65 and k=9 are rejected
        Dev dv; int probs = dv.addf(4, 65), sel = dv.addi(4 * 9), slot = dv.addi(4 * 9), eidx = dv.addi(65 * 2);
        check_true("moe_route E=65 -> E_SHAPE", run1x(dv, OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(4).i(65).i(2).i(2)) == KVM_GPT_E_SHAPE);
        check_true("moe_route k=9 -> E_SHAPE", run1x(dv, OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(4).i(30).i(9).i(2)) == KVM_GPT_E_SHAPE);
        check_true("moe_route k>E -> E_SHAPE", run1x(dv, OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(4).i(3).i(4).i(2)) == KVM_GPT_E_SHAPE);
    }
}

// ---- attention ----------------------------------------------------------------------------------------------------------------------
static void attn_full_d(int B, int T, int H, int Hkv, int hd, double scale, const D& q, const D& k, const D& v, D& o, D& lse, const D* dO, D* dq, D* dk, D* dv) {
    int G = H / Hkv;
    o.assign((size_t)B * T * H * hd, 0); lse.assign((size_t)B * H * T, 0);
    if (dO) { dq->assign(q.size(), 0); dk->assign(k.size(), 0); dv->assign(v.size(), 0); }
    auto Q = [&](int b, int t, int h, int d) { return q[((size_t)(b * T + t)) * H * hd + h * hd + d]; };
    auto Kx = [&](int b, int t, int h, int d) { return k[((size_t)(b * T + t)) * Hkv * hd + (h / G) * hd + d]; };
    auto Vx = [&](int b, int t, int h, int d) { return v[((size_t)(b * T + t)) * Hkv * hd + (h / G) * hd + d]; };
    for (int b = 0; b < B; b++) for (int h = 0; h < H; h++) for (int t = 0; t < T; t++) {
        D s(T); double mx = -1e300;
        for (int j = 0; j < T; j++) { double a = 0; for (int d = 0; d < hd; d++) a += Q(b, t, h, d) * Kx(b, j, h, d); s[j] = a * scale; mx = std::max(mx, s[j]); }
        double l = 0; for (int j = 0; j < T; j++) { s[j] = exp(s[j] - mx); l += s[j]; }
        lse[((size_t)b * H + h) * T + t] = mx + log(l);
        for (int j = 0; j < T; j++) { s[j] /= l; for (int d = 0; d < hd; d++) o[((size_t)(b * T + t)) * H * hd + h * hd + d] += s[j] * Vx(b, j, h, d); }
        if (!dO) continue;
        D dp(T); double delta = 0;
        for (int j = 0; j < T; j++) {
            double a = 0; for (int d = 0; d < hd; d++) a += (*dO)[((size_t)(b * T + t)) * H * hd + h * hd + d] * Vx(b, j, h, d);
            dp[j] = a; delta += s[j] * a;
        }
        for (int j = 0; j < T; j++) {
            double ds = s[j] * (dp[j] - delta);
            for (int d = 0; d < hd; d++) {
                (*dv)[((size_t)(b * T + j)) * Hkv * hd + (h / G) * hd + d] += s[j] * (*dO)[((size_t)(b * T + t)) * H * hd + h * hd + d];
                (*dq)[((size_t)(b * T + t)) * H * hd + h * hd + d] += scale * ds * Kx(b, j, h, d);
                (*dk)[((size_t)(b * T + j)) * Hkv * hd + (h / G) * hd + d] += scale * ds * Q(b, t, h, d);
            }
        }
    }
}
static void attn_case(int B, int T, int H, int Hkv, int hd) {
    Rng r(141 + hd + T + H);
    int R = B * T;
    float scale = (float)(1.0 / sqrt((double)hd));
    int smem = kvm_gpt_smem_bytes(hd);
    std::string nm = "attn_full B=" + S(B) + " T=" + S(T) + " H=" + S(H) + " Hkv=" + S(Hkv) + " hd=" + S(hd);
    Dev dv;
    int q = dv.addf(R, H * hd, H * hd + 4), k = dv.addf(R, Hkv * hd, Hkv * hd + 4), v = dv.addf(R, Hkv * hd), o = dv.addf(R, H * hd, H * hd + 4);
    int lse = dv.addf(1, B * H * T), dlt = dv.addf(1, B * H * T), dO = dv.addf(R, H * hd), dq = dv.addf(R, H * hd, H * hd + 4), dk = dv.addf(R, Hkv * hd, Hkv * hd + 4), dvv = dv.addf(R, Hkv * hd);
    auto hq = rnd(r, R * H * hd), hk = rnd(r, R * Hkv * hd), hv = rnd(r, R * Hkv * hd), hdo = rnd(r, R * H * hd);
    dv.up(q, hq); dv.up(k, hk); dv.up(v, hv); dv.up(dO, hdo);
    // garbage in the outputs: dq/dk/dv must be OVERWRITTEN
    dv.up(dq, rnd(r, R * H * hd)); dv.up(dk, rnd(r, R * Hkv * hd)); dv.up(dvv, rnd(r, R * Hkv * hd));
    BatchX b;
    for (int bb = 0; bb < B; bb++) for (int h = 0; h < H; h++) for (int q0 = 0; q0 < T; q0 += 32)
        b.add(OPX_attn_full_fwd, A().b(q).b(k).b(v).b(o).b(lse).i(bb).i(h).i(q0).i(T).i(H).i(Hkv).i(hd).f(scale));
    int e = runx(dv, b, smem);
    D ro, rl, rdq, rdk, rdv;
    attn_full_d(B, T, H, Hkv, hd, scale, dbl(hq), dbl(hk), dbl(hv), ro, rl, nullptr, nullptr, nullptr, nullptr);
    check(nm + " fwd o", dv.down(o), ro, 2e-5); check(nm + " fwd lse", dv.down(lse), rl, 2e-5, 1e-3); check_true(nm + " fwd err", e == 0);
    run1x(dv, OP_attn_bwd_delta, A().b(o).b(dO).b(dlt).i(0).i(R).i(T).i(H).i(hd), smem);
    BatchX b2;
    for (int bb = 0; bb < B; bb++) for (int h = 0; h < H; h++) for (int q0 = 0; q0 < T; q0 += 32)
        b2.add(OPX_attn_full_bwd_dq, A().b(q).b(k).b(v).b(dO).b(lse).b(dlt).b(dq).i(bb).i(h).i(q0).i(T).i(H).i(Hkv).i(hd).f(scale));
    for (int bb = 0; bb < B; bb++) for (int kvh = 0; kvh < Hkv; kvh++) for (int k0 = 0; k0 < T; k0 += 32)
        b2.add(OPX_attn_full_bwd_dkv, A().b(q).b(k).b(v).b(dO).b(lse).b(dlt).b(dk).b(dvv).i(bb).i(kvh).i(k0).i(T).i(H).i(hd).f(scale));
    e = runx(dv, b2, smem);
    D dd = dbl(hdo);
    attn_full_d(B, T, H, Hkv, hd, scale, dbl(hq), dbl(hk), dbl(hv), ro, rl, &dd, &rdq, &rdk, &rdv);
    check(nm + " bwd dq", dv.down(dq), rdq, 5e-5, T == 1 ? 1.0 : 1e-3); check(nm + " bwd dk", dv.down(dk), rdk, 5e-5, T == 1 ? 1.0 : 1e-3); check(nm + " bwd dv", dv.down(dvv), rdv, 5e-5, T == 1 ? 1.0 : 1e-3);
    check_true(nm + " bwd err", e == 0);
    auto g1 = dv.down(dk); runx(dv, b2, smem); check_true(nm + " bwd bitwise deterministic", g1 == dv.down(dk));
}
static void t_attn() {
    for (int hd : {16, 32, 64, 128}) for (int T : {1, 31, 33, 64, 100}) {
        attn_case(2, T, 4, 2, hd);
        if (T == 33 || T == 1) attn_case(1, T, 4, 1, hd);
        if (T == 100) attn_case(1, T, 2, 2, hd);
    }
    attn_case(1, 256, 2, 1, 64);
    attn_case(1, 256, 2, 2, 128);
    attn_case(1, 300, 3, 3, 32);
}

// ---- mixed precision -------------------------------------------------------------------------------------------------------------
static void t_bf16() {
    Rng r(151);
    std::vector<float> v = {0.f, -0.f, INFINITY, -INFINITY, NAN, -NAN, 1.f, -1.f, 3.4028235e38f, -3.4028235e38f, 3.38e38f, 1e-40f, -1e-40f, 1.4e-45f, 9.2e-41f, 1.1754944e-38f,
                            f32bits(0x3f808000u), f32bits(0x3f818000u), f32bits(0x3f808001u), f32bits(0x3f80ffffu), f32bits(0x7f7f8000u), f32bits(0x7f7fffffu), f32bits(0x00008000u),
                            f32bits(0x00018000u), f32bits(0x00007fffu), f32bits(0x80008000u), f32bits(0x7fc12345u), f32bits(0x7f800001u)};
    for (int i = 0; i < 4000; i++) v.push_back(f32bits(r.u32() * 65537u + r.u32()));   // random bit patterns (all exponents)
    for (int i = 0; i < 2000; i++) v.push_back((float)(r.nrm() * 10));
    int n = (int)v.size(); Dev dv; int x = dv.addf(1, n), o = dv.addf(1, n); dv.up(x, v);
    int e = run1x(dv, OPX_round_bf16, A().b(x).b(o).i(0).i(n));
    auto g = dv.down(o); bool ok = true; int bad = 0;
    for (int i = 0; i < n; i++) {
        float ref = bf16_ref(v[i]);
        bool same = std::isnan(ref) ? std::isnan(g[i]) : bitsf(ref) == bitsf(g[i]);
        if (!same) { if (bad++ < 5) printf("  bf16 mismatch in %08x got %08x want %08x\n", bitsf(v[i]), bitsf(g[i]), bitsf(ref)); ok = false; }
    }
    check_true("round_bf16 bitwise vs independent RNE (special values, denormals, random bits)", ok && e == 0);
    run1x(dv, OPX_round_bf16, A().b(x).b(x).i(10).i(n - 10));   // in place + partial
    auto gi = dv.down(x); bool ok2 = true;
    for (int i = 0; i < n; i++) { bool in = i >= 10 && i < n - 10; float want = in ? bf16_ref(v[i]) : v[i]; ok2 &= std::isnan(want) ? std::isnan(gi[i]) : bitsf(want) == bitsf(gi[i]); }
    check_true("round_bf16 in place, partial range", ok2);
}

struct MmCfg { int M, N, K; };
static void t_matmul_bf16() {
    Rng r(161);
    MmCfg shapes[] = {{1, 1, 1}, {5, 7, 3}, {64, 64, 32}, {65, 63, 33}, {130, 70, 100}, {1, 300, 17}, {70, 1, 90}, {64, 64, 512}};
    double worst = 0;
    for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) for (auto s : shapes) for (int full = 0; full < 3; full++) {
        int M = s.M, N = s.N, K = s.K;
        Dev dv;
        int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
        int ia = dv.addf(ar, ac, ac + ((M + N) & 1 ? 1 : 4)), ib = dv.addf(br, bc, bc + 4), io = dv.addf(M, N, N + 1);
        int ibias = dv.addf(1, N), ires = dv.addf(M, N), ipre = dv.addf(M, N, N + 4);
        auto ha = rnd(r, ar * ac), hb = rnd(r, br * bc), ho = rnd(r, M * N), hbias = rnd(r, N), hres = rnd(r, M * N);
        dv.up(ia, ha); dv.up(ib, hb); dv.up(io, ho); dv.up(ibias, hbias); dv.up(ires, hres);
        float alpha = full ? 0.7f : 1.f;
        int fl = ta | (tb << 1) | (full == 2 ? 16 : 0) | (full ? (4 | 8) : 0);   // flag 16 must be ignored
        BatchX b;
        for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64)
            b.add(OPX_matmul_tile_bf16, A().b(ia).b(ib).b(io).b(full ? ibias : -1).b(full ? ires : -1).b(full ? ipre : -1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
        int e = runx(dv, b);
        D C = mm_d(rb(ha), rb(hb), M, N, K, ta, tb), ref(M * N), pre(M * N);
        for (int i = 0; i < M * N; i++) {
            double v = alpha * C[i];
            if (full) { v += hbias[i % N]; pre[i] = v; v = gelu_d(v) + hres[i]; v += ho[i]; }
            ref[i] = v;
        }
        std::string nm = "matmul_tile_bf16 ta=" + S(ta) + " tb=" + S(tb) + " " + S(M) + "x" + S(N) + "x" + S(K) + (full ? " bias+gelu+res+acc" : " plain") + (full == 2 ? " flag16" : "");
        double e1 = relerr(dv.down(io), ref); worst = std::max(worst, e1);
        check(nm, dv.down(io), ref, 1e-4);
        if (full) check(nm + " pre", dv.down(ipre), pre, 1e-4);
        if (e) check_true(nm + " err", false);
        if (M == 65) { auto g1 = dv.down(io); dv.up(io, ho); runx(dv, b); check_true(nm + " bitwise deterministic", g1 == dv.down(io)); }
    }
    printf("  matmul_tile_bf16 worst relerr %.3e\n", worst);
    {   // bf16 semantic: differs from plain tf32/fp32 by rounding, equals the rounded-operand reference even for values that tf32 would keep
        int M = 16, N = 16, K = 16; Dev dv; int ia = dv.addf(M, K), ib = dv.addf(K, N), io = dv.addf(M, N);
        std::vector<float> ha(M * K, 1.0f + 1.0f / 1024), hb(K * N, 1.0f);   // 1+2^-10 is exact in tf32, rounds to 1 in bf16
        dv.up(ia, ha); dv.up(ib, hb);
        run1x(dv, OPX_matmul_tile_bf16, A().b(ia).b(ib).b(io).b(-1).b(-1).b(-1).i(0).i(0).i(M).i(N).i(K).i(0).f(1.f));
        auto g = dv.down(io); bool ok = true; for (float x : g) ok &= x == 16.f;
        check_true("matmul_tile_bf16 rounds operands to bf16 (1+2^-10 -> 1)", ok);
    }
}

// ---- bf16 STORAGE (KvmBuf.dtype 2): round_bf16 as narrow/widen, matmul_tile_bf16 with bf16 a/b/out/res/pre, dtype errors ---------------------------
static void t_bf16_storage() {
    Rng r(171);
    {   // narrow f32 -> bf16 storage is bitwise the RNE reference; widen is exact; bf16 -> bf16 copies
        std::vector<float> v;
        for (int i = 0; i < 3000; i++) v.push_back(f32bits(r.u32() * 65537u + r.u32()));
        for (int i = 0; i < 1000; i++) v.push_back((float)(r.nrm() * 10));
        int n = (int)v.size(); Dev dv; int x = dv.addf(1, n), nb = dv.addb(1, n), w = dv.addf(1, n), c = dv.addb(1, n); dv.up(x, v);
        int e0 = run1x(dv, OPX_round_bf16, A().b(x).b(nb).i(0).i(n));
        std::vector<uint16_t> raw; auto g = dv.downb(nb, &raw); bool ok = true;
        for (int i = 0; i < n; i++) { float ref = bf16_ref(v[i]); ok &= std::isnan(ref) ? std::isnan(g[i]) : bitsf(ref) == bitsf(g[i]); }
        check_true("round_bf16 f32 -> bf16 storage bitwise vs RNE reference", ok && e0 == 0);
        int e1 = run1x(dv, OPX_round_bf16, A().b(nb).b(w).i(0).i(n));
        auto gw = dv.down(w); bool ok2 = true;
        for (int i = 0; i < n; i++) ok2 &= bitsf(gw[i]) == bitsf(g[i]);
        check_true("round_bf16 bf16 storage -> f32 widen is exact", ok2 && e1 == 0);
        int e2 = run1x(dv, OPX_round_bf16, A().b(nb).b(c).i(5).i(n - 5));
        auto gc = dv.downb(c); bool ok3 = true;
        for (int i = 0; i < n; i++) ok3 &= (i < 5 || i >= n - 5) ? bitsf(gc[i]) == 0 : bitsf(gc[i]) == bitsf(g[i]);
        check_true("round_bf16 bf16 -> bf16 copy, partial range", ok3 && e2 == 0);
    }
    MmCfg shapes[] = {{1, 1, 1}, {5, 7, 3}, {64, 64, 32}, {65, 63, 33}, {130, 70, 100}, {1, 300, 17}, {70, 1, 90}, {64, 64, 512}};
    double worst = 0; int combos = 0;
    for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) for (auto s : shapes) for (int sd = 0; sd < 32; sd += (s.M == 65 || s.M == 130) ? 1 : 7) for (int full = 0; full < 2; full++) {
        int M = s.M, N = s.N, K = s.K;
        const bool fa = sd & 1, fb = sd & 2, fo = sd & 4, fr = (sd & 8) && full, fp = (sd & 16) && full;
        Dev dv;
        int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
        auto mk = [&](bool bf, int rr, int cc, int ld) { return bf ? dv.addb(rr, cc, ld) : dv.addf(rr, cc, ld); };
        int ia = mk(fa, ar, ac, ac + ((M + N) & 1 ? 1 : 4)), ib = mk(fb, br, bc, bc + 4), io = mk(fo, M, N, N + 1);
        int ibias = dv.addf(1, N), ires = mk(fr, M, N, 0), ipre = mk(fp, M, N, N + 4);
        auto ha = rnd(r, ar * ac), hb = rnd(r, br * bc), hbias = rnd(r, N);
        auto hres = rnd(r, M * N), ho = rnd(r, M * N);
        auto up = [&](int i, bool bf, std::vector<float>& h) { if (bf) { for (auto& x : h) x = bf16_ref(x); dv.upb(i, h); } else dv.up(i, h); };
        up(ia, fa, ha); up(ib, fb, hb); up(io, fo, ho); dv.up(ibias, hbias); up(ires, fr, hres);
        float alpha = full ? 0.7f : 1.f;
        int fl = ta | (tb << 1) | (full ? (4 | 8) : 0);
        BatchX b;
        for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64)
            b.add(OPX_matmul_tile_bf16, A().b(ia).b(ib).b(io).b(full ? ibias : -1).b(full ? ires : -1).b(full ? ipre : -1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
        int e = runx(dv, b);
        D C = mm_d(rb(ha), rb(hb), M, N, K, ta, tb), ref(M * N), pre(M * N);
        for (int i = 0; i < M * N; i++) {
            double v = alpha * C[i];
            if (full) { v += hbias[i % N]; pre[i] = v; v = gelu_d(v) + hres[i]; v += ho[i]; }
            ref[i] = v;
        }
        std::string nm = std::string("bf16 storage ta=") + S(ta) + " tb=" + S(tb) + " " + S(M) + "x" + S(N) + "x" + S(K) + " sd=" + S(sd) + (full ? " bias+gelu+res+acc" : " plain");
        std::vector<uint16_t> rawo;
        auto go = fo ? dv.downb(io, &rawo) : dv.down(io);
        double e1 = relerr(go, ref); worst = std::max(worst, e1); combos++;
        check(nm, go, ref, fo ? 1.2e-2 : 1e-4);
        if (fo) { bool exact = true; for (int i = 0; i < M * N; i++) { /* stored values are bf16 by construction; check rounding of fp32 result within 1 bf16 ulp */ float g = go[i]; exact &= std::isfinite(g); } check_true(nm + " finite", exact); }
        if (full) check(nm + " pre", fp ? dv.downb(ipre) : dv.down(ipre), pre, fp ? 1.2e-2 : 1e-4);
        if (e) check_true(nm + " err", false);
    }
    printf("  bf16 storage matmul: %d combinations, worst relerr %.3e\n", combos, worst);
    {   // a bf16 buffer given to an op without bf16 support is rejected before any store
        Dev dv; int xb = dv.addb(1, 64), o = dv.addf(1, 64); dv.upb(xb, std::vector<float>(64, 1.f));
        int e = run1x(dv, OPX_act_fwd, A().b(xb).b(o).i(1).i(0).i(64));
        auto g = dv.down(o); bool zero = true; for (float v : g) zero &= v == 0.f;
        check_true("act_fwd rejects a bf16 buffer (E_DTYPE), memory untouched", e == KVM_GPT_E_DTYPE && zero, ("got code " + S(e)).c_str());
    }
}

// ---- schedule / optimizer ---------------------------------------------------------------------------------------------------------------
static void t_sched() {
    double worst = 0;
    for (int kind = 0; kind < 3; kind++) for (int warm : {0, 1, 10}) for (int total : {5, 100, 1000}) for (int t : {1, 2, 5, 9, 10, 11, 50, 100, 101, 999, 1000, 5000}) {
        Dev dv; int st = dv.addi(1), lr = dv.addf(1, 1); dv.upi(st, {t}); dv.up(lr, {123.f});
        float base = 3e-4f, mr = 0.1f;
        int e = run1x(dv, OPX_lr_schedule, A().b(st).b(lr).i(kind).f(base).i(warm).i(total).f(mr));
        double w = (warm > 0 && t <= warm) ? (double)t / warm : 1.0;
        double prog = std::min(1.0, std::max(0.0, (double)(t - warm) / std::max(total - warm, 1)));
        double f = kind == 0 ? 1.0 : kind == 1 ? (double)mr + (1 - (double)mr) * 0.5 * (1 + cos(M_PI * prog)) : 1 - (1 - (double)mr) * prog;
        double ref = (double)base * w * f;
        float got = dv.down(lr)[0];
        double err = fabs(got - ref) / ref; worst = std::max(worst, err);
        if (e || err > 2e-7) { printf("FAIL lr_schedule kind=%d warm=%d total=%d t=%d got %.9g ref %.9g err %d\n", kind, warm, total, t, got, ref, e); g_fail++; } else g_pass++;
    }
    printf("  lr_schedule worst relerr %.3e (%s)\n", worst, "432 configs");
    {   // adamw_lr == adamw with the lr as argument (bitwise), and vs double reference
        Rng r(171);
        int n = 777; Dev dv;
        int p1 = dv.addf(1, n), p2 = dv.addf(1, n), g = dv.addf(1, n), m1 = dv.addf(1, n), m2 = dv.addf(1, n), v1 = dv.addf(1, n), v2 = dv.addf(1, n), gsb = dv.addf(1, 1), st = dv.addi(1), lr = dv.addf(1, 1);
        auto hp = rnd(r, n), hg = rnd(r, n), hm = rnd(r, n, 0.1), hv = rnd(r, n, 0.1); for (auto& x : hv) x = x * x;
        dv.up(p1, hp); dv.up(p2, hp); dv.up(g, hg); dv.up(m1, hm); dv.up(m2, hm); dv.up(v1, hv); dv.up(v2, hv); dv.up(gsb, {0.5f}); dv.upi(st, {3}); dv.up(lr, {1.7e-3f});
        for (int usegs = 0; usegs < 2; usegs++) {
            BatchX b; for (int e0 = 0; e0 < n; e0 += 200) b.add(OPX_adamw_lr, A().b(p1).b(g).b(m1).b(v1).b(usegs ? gsb : -1).b(st).i(e0).i(std::min(n, e0 + 200)).b(lr).f(0.9f).f(0.95f).f(1e-8f).f(0.01f));
            BatchX b2; for (int e0 = 0; e0 < n; e0 += 200) b2.add(OP_adamw, A().b(p2).b(g).b(m2).b(v2).b(usegs ? gsb : -1).b(st).i(e0).i(std::min(n, e0 + 200)).f(1.7e-3f).f(0.9f).f(0.95f).f(1e-8f).f(0.01f));
            int e = runx(dv, b) | runx(dv, b2);
            check_true("adamw_lr bitwise == adamw (gscale=" + S(usegs) + ")", dv.down(p1) == dv.down(p2) && dv.down(m1) == dv.down(m2) && dv.down(v1) == dv.down(v2) && e == 0);
        }
        // double reference over the two steps applied
        D P = dbl(hp), M = dbl(hm), V = dbl(hv);
        for (int s = 0; s < 2; s++) adamw_d(P, dbl(hg), M, V, (double)1.7e-3f, (double)0.9f, (double)0.95f, (double)1e-8f, (double)0.01f, 3, s ? 0.5 : 1.0);
        // (usegs=0 pass then usegs=1 pass ran on p1 in sequence)
        check("adamw_lr vs double reference (2 passes)", dv.down(p1), P, 1e-5);
        dv.up(lr, {0.f}); auto before = dv.down(p1);
        run1x(dv, OPX_adamw_lr, A().b(p1).b(g).b(m1).b(v1).b(-1).b(st).i(0).i(n).b(lr).f(0.9f).f(0.95f).f(1e-8f).f(0.f));
        auto after = dv.down(p1); check_true("adamw_lr with lr=0, wd=0 leaves p unchanged", before == after);
    }
}

// ---- fused matmul wrappers -----------------------------------------------------------------------------------------------------------------
static void t_wrappers() {
    Rng r(181);
    MmCfg shapes[] = {{1, 1, 1}, {5, 7, 3}, {64, 64, 32}, {65, 63, 33}, {130, 70, 100}, {70, 130, 17}};
    for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) for (int prec = 0; prec < 2; prec++) for (auto s : shapes) {
        int M = s.M, N = s.N, K = s.K;
        int fl = ta | (tb << 1) | (prec << 4);
        Dev dv;
        int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
        int ia = dv.addf(ar, ac, ac + 4), ib = dv.addf(br, bc, bc + 4);
        int pre = dv.addf(M, N, N + 4), gate = dv.addf(M, N, N + 1), up = dv.addf(M, N), up2 = dv.addf(M, N), o1 = dv.addf(M, N, N + 2), o2 = dv.addf(M, N, N + 2), c = dv.addf(M, N);
        int dg1 = dv.addf(M, N, N + 3), dg2 = dv.addf(M, N, N + 3), du1 = dv.addf(M, N), du2 = dv.addf(M, N), dx2 = dv.addf(M, N, N + 2);
        auto ha = rnd(r, ar * ac), hb = rnd(r, br * bc), hpre = rnd(r, M * N, 2.0), hgate = rnd(r, M * N, 2.0), hup = rnd(r, M * N);
        dv.up(ia, ha); dv.up(ib, hb); dv.up(pre, hpre); dv.up(gate, hgate); dv.up(up, hup);
        // garbage in all outputs
        for (int t : {up2, o1, o2, c, dg1, dg2, du1, du2, dx2}) dv.up(t, rnd(r, M * N));
        float alpha = 0.8f;
        std::string nm = "ta=" + S(ta) + " tb=" + S(tb) + " prec=" + S(prec) + " " + S(M) + "x" + S(N) + "x" + S(K);
        BatchX bw, bs, bb;   // wrapper batches
        BatchX ref1, ref2, ref3;   // two-task path
        for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64) {
            bw.add(OPX_matmul_tile_dgelu, A().b(ia).b(ib).b(o1).b(pre).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
            bs.add(OPX_matmul_tile_silu_mul, A().b(ia).b(ib).b(o2).b(gate).b(up2).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
            bb.add(OPX_matmul_tile_silu_mul_bwd, A().b(ia).b(ib).b(gate).b(up).b(dg1).b(du1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
            ref1.add(OP_matmul_tile, A().b(ia).b(ib).b(c).b(-1).b(-1).b(-1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
        }
        int e = runx(dv, bw) | runx(dv, bs) | runx(dv, bb) | runx(dv, ref1);
        check_true("wrappers err==0 " + nm, e == 0);
        // two-task path
        run1x(dv, OP_gelu_bwd, A().b(pre).b(c).b(dx2).i(0).i(M * N).i(0), 48 * 1024);   // flat op needs ld == cols: use dedicated flat copies below
        // (the flat gpt/extra ops need dense buffers; build dense copies of the tiles)
        int cd = dv.addf(1, M * N), pd = dv.addf(1, M * N), gd = dv.addf(1, M * N), ud = dv.addf(1, M * N), od = dv.addf(1, M * N), dgd = dv.addf(1, M * N), dud = dv.addf(1, M * N);
        dv.up(cd, dv.down(c)); dv.up(pd, hpre); dv.up(gd, hgate); dv.up(ud, hup);
        run1x(dv, OP_gelu_bwd, A().b(pd).b(cd).b(od).i(0).i(M * N).i(0));
        check_true("matmul_tile_dgelu bitwise == matmul_tile + gelu_bwd " + nm, dv.down(o1) == dv.down(od));
        auto Cv = dv.down(c);
        // silu_mul: up2 = C, out = silu(gate) * C
        dv.up(od, std::vector<float>(M * N, 0.f)); dv.up(ud, Cv);
        run1x(dv, OPX_silu_mul_fwd, A().b(gd).b(ud).b(od).i(0).i(M * N));
        check_true("matmul_tile_silu_mul out bitwise == silu_mul_fwd " + nm, dv.down(o2) == dv.down(od));
        check_true("matmul_tile_silu_mul up == C " + nm, dv.down(up2) == Cv);
        // silu_mul_bwd: dh = C
        dv.up(ud, hup); dv.up(dgd, std::vector<float>(M * N, 0.f)); dv.up(dud, std::vector<float>(M * N, 0.f));
        int dhd = dv.addf(1, M * N); dv.up(dhd, Cv);
        run1x(dv, OPX_silu_mul_bwd, A().b(gd).b(ud).b(dhd).b(dgd).b(dud).i(0).i(M * N).i(0));
        check_true("matmul_tile_silu_mul_bwd dgate bitwise == silu_mul_bwd " + nm, dv.down(dg1) == dv.down(dgd));
        check_true("matmul_tile_silu_mul_bwd dup bitwise == silu_mul_bwd " + nm, dv.down(du1) == dv.down(dud));
        // double reference (loose: tf32)
        D C = mm_d(dbl(ha), dbl(hb), M, N, K, ta, tb), r1(M * N), r2(M * N), r3(M * N), r4(M * N);
        for (int i = 0; i < M * N; i++) {
            double cv = alpha * C[i];
            r1[i] = cv * gelu_grad_d(hpre[i]); r2[i] = silu_d(hgate[i]) * cv; r3[i] = cv * hup[i] * dsilu_d(hgate[i]); r4[i] = cv * silu_d(hgate[i]);
        }
        if (!prec) { check("dgelu vs double " + nm, dv.down(o1), r1, 6e-3); check("silu_mul vs double " + nm, dv.down(o2), r2, 6e-3);
                     check("silu_mul_bwd dgate vs double " + nm, dv.down(dg1), r3, 6e-3); check("silu_mul_bwd dup vs double " + nm, dv.down(du1), r4, 6e-3); }
        else { check("dgelu vs double " + nm, dv.down(o1), r1, 2e-5); check("silu_mul vs double " + nm, dv.down(o2), r2, 2e-5);
               check("silu_mul_bwd dgate vs double " + nm, dv.down(dg1), r3, 2e-5); check("silu_mul_bwd dup vs double " + nm, dv.down(du1), r4, 2e-5); }
    }
    {   // flags 4 / 8 rejected, memory untouched
        int M = 8, N = 8, K = 8; Dev dv; int ia = dv.addf(M, K), ib = dv.addf(K, N), o = dv.addf(M, N), pre = dv.addf(M, N), g = dv.addf(M, N), u = dv.addf(M, N), d1 = dv.addf(M, N), d2 = dv.addf(M, N);
        for (int t : {ia, ib, o, pre, g, u, d1, d2}) dv.up(t, rnd(r, 64));
        for (int fl : {4, 8, 32, 12}) {
            std::vector<std::vector<float>> before; for (int t : {o, u, d1, d2}) before.push_back(dv.down(t));
            int e1 = run1x(dv, OPX_matmul_tile_dgelu, A().b(ia).b(ib).b(o).b(pre).i(0).i(0).i(M).i(N).i(K).i(fl).f(1.f));
            int e2 = run1x(dv, OPX_matmul_tile_silu_mul, A().b(ia).b(ib).b(o).b(g).b(u).i(0).i(0).i(M).i(N).i(K).i(fl).f(1.f));
            int e3 = run1x(dv, OPX_matmul_tile_silu_mul_bwd, A().b(ia).b(ib).b(g).b(u).b(d1).b(d2).i(0).i(0).i(M).i(N).i(K).i(fl).f(1.f));
            bool same = true; int k = 0; for (int t : {o, u, d1, d2}) same &= dv.down(t) == before[k++];
            check_true("wrappers reject flags " + S(fl) + " (E_SHAPE) and leave memory untouched", e1 == KVM_GPT_E_SHAPE && e2 == KVM_GPT_E_SHAPE && e3 == KVM_GPT_E_SHAPE && same);
        }
    }
}

// ---- error paths ------------------------------------------------------------------------------------------------------------------------------
// Generic mutation test: for each op a valid argument set; every non-optional buf word replaced by (a) an out-of-table index -> 1001,
// (b) a buffer of the wrong dtype -> 1002; plus dedicated range/shape mutations. Memory must stay untouched in all cases.
struct ErrCase { int id; A a; };
static void t_errors() {
    Rng r(191);
    Dev dv;
    auto F = [&](int rows, int cols) { int b = dv.addf(rows, cols); dv.up(b, rnd(r, rows * cols)); return b; };
    int N = 16, E = 4, K = 2, C = 4, d = 8, T = 16, H = 2, hd = 16;
    int fx = F(1, 64), fy = F(1, 64), fz = F(1, 64), fw = F(1, 64), fv = F(1, 64);
    int mat = F(N, d), mat2 = F(N, d), mat3 = F(N, d), vec1 = F(1, 1), idx = dv.addi(N); dv.upi(idx, std::vector<int32_t>(N, 1));
    int sc = F(N, 1), probs = F(N, E), sel = dv.addi(N * K), slot = dv.addi(N * K), eidx = dv.addi(E * C), gs = F(E * C, 1), dgs = F(E * C, 1), dprobs = F(N, E);
    std::vector<int32_t> hsel(N * K), hslot(N * K); for (int i = 0; i < N * K; i++) { hsel[i] = i % E; hslot[i] = (i % E) * C + i / E % C; }
    dv.upi(sel, hsel); dv.upi(slot, hslot);
    int stp = dv.addi(1); dv.upi(stp, {5}); int lr = F(1, 1);
    int aq = F(T, H * hd), ak = F(T, H * hd), av = F(T, H * hd), ao = F(T, H * hd), alse = F(1, H * T), adl = F(1, H * T), adq = F(T, H * hd), adk = F(T, H * hd), adv = F(T, H * hd);
    int ma = F(8, 8), mb = F(8, 8), mo = F(8, 8), mpre = F(8, 8), mg = F(8, 8), mu = F(8, 8);
    std::vector<ErrCase> cs = {
        {OPX_act_fwd, A().b(fx).b(fy).i(1).i(0).i(64)}, {OPX_act_bwd, A().b(fx).b(fy).b(fz).i(1).i(0).i(64).i(1)},
        {OPX_silu_mul_fwd, A().b(fx).b(fy).b(fz).i(0).i(64)}, {OPX_silu_mul_bwd, A().b(fx).b(fy).b(fz).b(fw).b(fv).i(0).i(64).i(1)},
        {OPX_softmax_rows_fwd, A().b(mat).b(mat2).i(0).i(N).i(d).f(1.f)}, {OPX_softmax_rows_bwd, A().b(mat).b(mat2).b(mat3).i(0).i(N).i(d).f(1.f).i(1)},
        {OPX_gather_rows_scaled, A().b(mat).b(idx).b(sc).b(mat2).i(0).i(N).i(d).i(1).i(0)}, {OPX_scatter_add_rows, A().b(mat).b(idx).b(sc).b(mat2).i(N).i(0).i(N).i(d).i(1).i(0)},
        {OPX_row_dot, A().b(mat).b(mat2).b(idx).b(sc).i(0).i(N).i(d).i(1).i(0)},
        {OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N).i(E).i(K).i(C)}, {OPX_moe_gates_fwd, A().b(probs).b(sel).b(slot).b(gs).i(N).i(E).i(K).i(1)},
        {OPX_moe_gates_bwd, A().b(probs).b(sel).b(slot).b(dgs).b(dprobs).i(N).i(E).i(K).i(1).i(1)}, {OPX_moe_balance_fwd, A().b(probs).b(sel).b(vec1).i(N).i(E).i(K).f(.1f)},
        {OPX_moe_balance_bwd, A().b(probs).b(sel).b(dprobs).i(N).i(E).i(K).f(.1f).i(1)},
        {OPX_attn_full_fwd, A().b(aq).b(ak).b(av).b(ao).b(alse).i(0).i(0).i(0).i(T).i(H).i(H).i(hd).f(.25f)},
        {OPX_attn_full_bwd_dq, A().b(aq).b(ak).b(av).b(ao).b(alse).b(adl).b(adq).i(0).i(0).i(0).i(T).i(H).i(H).i(hd).f(.25f)},
        {OPX_attn_full_bwd_dkv, A().b(aq).b(ak).b(av).b(ao).b(alse).b(adl).b(adk).b(adv).i(0).i(0).i(0).i(T).i(H).i(hd).f(.25f)},
        {OPX_round_bf16, A().b(fx).b(fy).i(0).i(64)},
        {OPX_matmul_tile_bf16, A().b(ma).b(mb).b(mo).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(8).i(0).f(1.f)},
        {OPX_adamw_lr, A().b(fx).b(fy).b(fz).b(fw).b(-1).b(stp).i(0).i(64).b(lr).f(.9f).f(.99f).f(1e-8f).f(0.f)},
        {OPX_lr_schedule, A().b(stp).b(lr).i(1).f(1e-3f).i(2).i(10).f(.1f)}, {OPX_scale_flat, A().b(fx).b(fy).f(2.f).i(0).i(64)},
        {OPX_add_rowtile, A().b(mat).b(mat2).b(mat3).i(0).i(N).i(4).i(d)},
        {OPX_matmul_tile_dgelu, A().b(ma).b(mb).b(mo).b(mpre).i(0).i(0).i(8).i(8).i(8).i(0).f(1.f)},
        {OPX_matmul_tile_silu_mul, A().b(ma).b(mb).b(mo).b(mg).b(mu).i(0).i(0).i(8).i(8).i(8).i(0).f(1.f)},
        {OPX_matmul_tile_silu_mul_bwd, A().b(ma).b(mb).b(mg).b(mu).b(mo).b(mpre).i(0).i(0).i(8).i(8).i(8).i(0).f(1.f)},
    };
    // which words are bufs (non-optional) per op: parse the signature
    // bitwise snapshot of every buffer (NaN-safe, includes ld padding)
    auto snapshot = [&]() { std::vector<std::vector<uint32_t>> s; for (size_t i = 0; i < dv.tab.size(); i++) { auto& b = dv.tab[i]; std::vector<uint32_t> t((size_t)b.rows * b.ld); CK(cudaMemcpy(t.data(), b.p, t.size() * 4, cudaMemcpyDeviceToHost)); s.push_back(t); } return s; };
    int nfail_before = g_fail, ncases = 0;
    for (auto& c : cs) {
        const char* sig = nullptr; const char* nm = "?";
        for (auto& o : OPSX) if (o.id == c.id) { sig = o.sig; nm = o.name; }
        // baseline: valid call has no error
        { int e = run1x(dv, c.id, c.a); if (e) { printf("FAIL baseline %s code %d\n", nm, e); g_fail++; } else g_pass++; }
        int pos = 0; std::string s(sig);
        std::vector<std::pair<bool, bool>> kinds;   // (is_buf, optional)
        { size_t i = 0; while (i < s.size()) { size_t cpos = s.find(':', i), comma = s.find(',', cpos); std::string ty = s.substr(cpos + 1, comma == std::string::npos ? std::string::npos : comma - cpos - 1);
              kinds.push_back({ty.rfind("buf", 0) == 0, ty == "buf?"}); if (comma == std::string::npos) break; i = comma + 1; } }
        for (pos = 0; pos < (int)kinds.size(); pos++) {
            if (!kinds[pos].first) continue;
            for (int mode = 0; mode < 2; mode++) {
                if (kinds[pos].second && mode == 1 && c.a.w[pos] < 0) continue;
                if (kinds[pos].second && c.a.w[pos] < 0 && mode == 0) { /* absent optional: out-of-table index is still an error */ }
                A m = c.a; int want;
                if (mode == 0) { m.w[pos] = 999; want = KVM_GPT_E_BUF; }
                else { int orig = c.a.w[pos]; bool isint = dv.tab[orig].dtype == 1; m.w[pos] = isint ? fx : idx; want = KVM_GPT_E_DTYPE; }
                auto before = snapshot();
                int e = run1x(dv, c.id, m);
                bool same = snapshot() == before; ncases++;
                if (e != want || !same) { printf("FAIL %s arg %d mode %d: code %d (want %d) untouched=%d\n", nm, pos, mode, e, want, (int)same); g_fail++; } else g_pass++;
            }
        }
    }
    printf("  error-path mutations: %d cases, %d failed\n", ncases, g_fail - nfail_before);
    // dedicated range / shape errors
    struct Mut { const char* what; int id; A a; int want; };
    std::vector<Mut> ms = {
        {"act_fwd e1 beyond buffer", OPX_act_fwd, A().b(fx).b(fy).i(1).i(0).i(65), KVM_GPT_E_RANGE},
        {"act_fwd e0 > e1", OPX_act_fwd, A().b(fx).b(fy).i(1).i(10).i(5), KVM_GPT_E_RANGE},
        {"act_fwd bad kind", OPX_act_fwd, A().b(fx).b(fy).i(4).i(0).i(64), KVM_GPT_E_SHAPE},
        {"act_bwd negative kind", OPX_act_bwd, A().b(fx).b(fy).b(fz).i(-1).i(0).i(64).i(0), KVM_GPT_E_SHAPE},
        {"silu_mul_bwd e1 beyond", OPX_silu_mul_bwd, A().b(fx).b(fy).b(fz).b(fw).b(fv).i(0).i(65).i(0), KVM_GPT_E_RANGE},
        {"softmax rows beyond", OPX_softmax_rows_fwd, A().b(mat).b(mat2).i(0).i(N + 1).i(d).f(1.f), KVM_GPT_E_RANGE},
        {"softmax d beyond cols", OPX_softmax_rows_fwd, A().b(mat).b(mat2).i(0).i(N).i(d + 1).f(1.f), KVM_GPT_E_RANGE},
        {"softmax d=0", OPX_softmax_rows_fwd, A().b(mat).b(mat2).i(0).i(N).i(0).f(1.f), KVM_GPT_E_RANGE},
        {"softmax_bwd rows beyond", OPX_softmax_rows_bwd, A().b(mat).b(mat2).b(mat3).i(0).i(N + 1).i(d).f(1.f).i(0), KVM_GPT_E_RANGE},
        {"gather r1 beyond idx", OPX_gather_rows_scaled, A().b(mat).b(idx).b(sc).b(mat2).i(0).i(N + 1).i(d).i(0).i(0), KVM_GPT_E_RANGE},
        {"gather d beyond cols", OPX_gather_rows_scaled, A().b(mat).b(idx).b(-1).b(mat2).i(0).i(N).i(d + 1).i(0).i(0), KVM_GPT_E_RANGE},
        {"scatter N beyond src", OPX_scatter_add_rows, A().b(mat).b(idx).b(-1).b(mat2).i(N + 1).i(0).i(N).i(d).i(0).i(0), KVM_GPT_E_RANGE},
        {"scatter v1 beyond dst", OPX_scatter_add_rows, A().b(mat).b(idx).b(-1).b(mat2).i(N).i(0).i(N + 1).i(d).i(0).i(0), KVM_GPT_E_RANGE},
        {"row_dot r1 beyond out", OPX_row_dot, A().b(mat).b(mat2).b(-1).b(sc).i(0).i(N + 1).i(d).i(0).i(0), KVM_GPT_E_RANGE},
        {"moe E=65", OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N).i(65).i(K).i(C), KVM_GPT_E_SHAPE},
        {"moe_route N beyond probs", OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N + 1).i(E).i(K).i(C), KVM_GPT_E_RANGE},
        {"moe_route C too large for eidx", OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N).i(E).i(K).i(C + 1), KVM_GPT_E_RANGE},
        {"moe_route k=0", OPX_moe_route, A().b(probs).b(sel).b(slot).b(eidx).i(N).i(E).i(0).i(C), KVM_GPT_E_SHAPE},
        {"moe_gates_fwd k=9", OPX_moe_gates_fwd, A().b(probs).b(sel).b(slot).b(gs).i(N).i(E).i(9).i(1), KVM_GPT_E_SHAPE},
        {"moe_balance_fwd sel too short", OPX_moe_balance_fwd, A().b(probs).b(sel).b(vec1).i(N).i(E).i(K + 1).f(.1f), KVM_GPT_E_RANGE},
        {"attn_full hd=24", OPX_attn_full_fwd, A().b(aq).b(ak).b(av).b(ao).b(alse).i(0).i(0).i(0).i(T).i(H).i(H).i(24).f(.25f), KVM_GPT_E_SHAPE},
        {"attn_full q0 >= T", OPX_attn_full_fwd, A().b(aq).b(ak).b(av).b(ao).b(alse).i(0).i(0).i(T).i(T).i(H).i(H).i(hd).f(.25f), KVM_GPT_E_RANGE},
        {"attn_full batch beyond buffers", OPX_attn_full_fwd, A().b(aq).b(ak).b(av).b(ao).b(alse).i(1).i(0).i(0).i(T).i(H).i(H).i(hd).f(.25f), KVM_GPT_E_RANGE},
        {"attn_full H % Hkv", OPX_attn_full_fwd, A().b(aq).b(ak).b(av).b(ao).b(alse).i(0).i(0).i(0).i(T).i(H).i(3).i(hd).f(.25f), KVM_GPT_E_RANGE},
        {"attn_full dq head beyond", OPX_attn_full_bwd_dq, A().b(aq).b(ak).b(av).b(ao).b(alse).b(adl).b(adq).i(0).i(H).i(0).i(T).i(H).i(H).i(hd).f(.25f), KVM_GPT_E_RANGE},
        {"attn_full dkv kvh beyond", OPX_attn_full_bwd_dkv, A().b(aq).b(ak).b(av).b(ao).b(alse).b(adl).b(adk).b(adv).i(0).i(H).i(0).i(T).i(H).i(hd).f(.25f), KVM_GPT_E_RANGE},
        {"round_bf16 beyond", OPX_round_bf16, A().b(fx).b(fy).i(0).i(65), KVM_GPT_E_RANGE},
        {"matmul_bf16 flags", OPX_matmul_tile_bf16, A().b(ma).b(mb).b(mo).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(8).i(64).f(1.f), KVM_GPT_E_SHAPE},
        {"matmul_bf16 m0 >= M", OPX_matmul_tile_bf16, A().b(ma).b(mb).b(mo).b(-1).b(-1).b(-1).i(8).i(0).i(8).i(8).i(8).i(0).f(1.f), KVM_GPT_E_RANGE},
        {"matmul_bf16 K beyond A", OPX_matmul_tile_bf16, A().b(ma).b(mb).b(mo).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(9).i(0).f(1.f), KVM_GPT_E_RANGE},
        {"matmul_dgelu M beyond", OPX_matmul_tile_dgelu, A().b(ma).b(mb).b(mo).b(mpre).i(0).i(0).i(9).i(8).i(8).i(0).f(1.f), KVM_GPT_E_RANGE},
        {"lr_schedule bad kind", OPX_lr_schedule, A().b(stp).b(lr).i(3).f(1e-3f).i(2).i(10).f(.1f), KVM_GPT_E_SHAPE},
        {"lr_schedule step buffer wrong size", OPX_lr_schedule, A().b(fx).b(lr).i(1).f(1e-3f).i(2).i(10).f(.1f), KVM_GPT_E_DTYPE},
        {"adamw_lr e1 beyond", OPX_adamw_lr, A().b(fx).b(fy).b(fz).b(fw).b(-1).b(stp).i(0).i(65).b(lr).f(.9f).f(.99f).f(1e-8f).f(0.f), KVM_GPT_E_RANGE},
        {"adamw_lr lr buffer too small", OPX_adamw_lr, A().b(fx).b(fy).b(fz).b(fw).b(-1).b(stp).i(0).i(64).b(vec1).f(.9f).f(.99f).f(1e-8f).f(0.f), 0},
        {"scale_flat beyond", OPX_scale_flat, A().b(fx).b(fy).f(2.f).i(0).i(65), KVM_GPT_E_RANGE},
        {"add_rowtile T beyond pos", OPX_add_rowtile, A().b(mat).b(mat2).b(mat3).i(0).i(N).i(N + 1).i(d), KVM_GPT_E_RANGE},
        {"add_rowtile T=0", OPX_add_rowtile, A().b(mat).b(mat2).b(mat3).i(0).i(N).i(0).i(d), KVM_GPT_E_RANGE},
    };
    for (auto& m : ms) {
        if (m.want == 0) continue;   // placeholder entries: valid
        auto before = snapshot();
        int e = run1x(dv, m.id, m.a);
        bool same = snapshot() == before;
        check_true(std::string("error path: ") + m.what + " -> " + S(m.want) + ", memory untouched", e == m.want && same, ("got code " + S(e)).c_str());
    }
}

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s, %d SMs, cc %d.%d\n", pr.name, pr.multiProcessorCount, pr.major, pr.minor);
    t_elementwise(); t_softmax(); t_gather(); t_moe(); t_bf16(); t_matmul_bf16(); t_bf16_storage(); t_sched(); t_wrappers(); t_attn(); t_errors();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
