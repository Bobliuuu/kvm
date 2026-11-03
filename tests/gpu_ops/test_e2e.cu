// End-to-end: a 2-layer GPT (vocab 64, d 32, 2 heads / 1 KV head (GQA), seq 16, batch 4, rope, pre-LN, tied head, GELU MLP)
// trained for 200 AdamW steps purely as a host-side sequence of the library's ops (one launch per op), against an independent
// double-precision CPU implementation of the same model. Checks: (1) CPU analytic gradients vs finite differences,
// (2) GPU gradients vs CPU gradients at step 0, (3) loss trajectory step for step, (4) loss falls, (5) bitwise reproducibility.
// Build: nvcc -O2 -arch=sm_80 -std=c++17 -I../../include -I. test_e2e.cu -o test_e2e
#include <algorithm>
#include <functional>
#include "harness.cuh"
#include "refs.hpp"

constexpr int V = 64, DM = 32, H = 2, HKV = 1, HD = 16, L = 2, T = 16, BT = 4, N = BT * T, F = 128, STEPS = 200;
constexpr double LR = 3e-3, B1 = 0.9, B2 = 0.99, EPS = 1e-8, WD = 0.01, CLIP = 1.0, LNEPS = 1e-5, THETA = 10000.0, SCALE = 0.25;
enum { WTE, WPE, LNFG, LNFB, LAYER0 };
enum { LN1G, LN1B, WQ, BQ, WK, BK, WV, BV, WO, BO, LN2G, LN2B, W1, B1_, W2, B2_, PER };
static int pidx(int l, int w) { return LAYER0 + l * PER + w; }
static const int NP = LAYER0 + L * PER;
static void shapes(std::vector<int>& R, std::vector<int>& C) {
    R.assign(NP, 0); C.assign(NP, 0);
    auto set = [&](int i, int r, int c) { R[i] = r; C[i] = c; };
    set(WTE, V, DM); set(WPE, T, DM); set(LNFG, 1, DM); set(LNFB, 1, DM);
    for (int l = 0; l < L; l++) {
        set(pidx(l, LN1G), 1, DM); set(pidx(l, LN1B), 1, DM); set(pidx(l, WQ), DM, H * HD); set(pidx(l, BQ), 1, H * HD);
        set(pidx(l, WK), DM, HKV * HD); set(pidx(l, BK), 1, HKV * HD); set(pidx(l, WV), DM, HKV * HD); set(pidx(l, BV), 1, HKV * HD);
        set(pidx(l, WO), H * HD, DM); set(pidx(l, BO), 1, DM); set(pidx(l, LN2G), 1, DM); set(pidx(l, LN2B), 1, DM);
        set(pidx(l, W1), DM, F); set(pidx(l, B1_), 1, F); set(pidx(l, W2), F, DM); set(pidx(l, B2_), 1, DM);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// CPU double reference model
static D rope_d(const D& x, int heads, bool inv) {
    D o = x;
    for (int r = 0; r < N; r++) for (int h = 0; h < heads; h++) for (int j = 0; j < HD / 2; j++) {
        double a = (r % T) * pow(THETA, -2.0 * j / HD), c = cos(a), s = (inv ? -1 : 1) * sin(a);
        double x1 = x[r * heads * HD + h * HD + j], x2 = x[r * heads * HD + h * HD + j + HD / 2];
        o[r * heads * HD + h * HD + j] = x1 * c - x2 * s; o[r * heads * HD + h * HD + j + HD / 2] = x1 * s + x2 * c;
    }
    return o;
}
static D lin(const D& x, const D& W, const D& b, int in, int out) {
    D y = mm_d(x, W, N, out, in, false, false);
    for (int r = 0; r < N; r++) for (int c = 0; c < out; c++) y[r * out + c] += b[c];
    return y;
}
struct LAct { D hin, n1, m1, r1, q, k, v, qr, kr, att, lse, h1, n2, m2, r2, u, act; };
// returns loss; fills grads if G != nullptr
static double cpu_model(const std::vector<D>& P, const std::vector<int>& ids, const std::vector<int>& tgt, std::vector<D>* G) {
    std::vector<LAct> A(L); D h(N * DM);
    for (int r = 0; r < N; r++) for (int c = 0; c < DM; c++) h[r * DM + c] = P[WTE][ids[r] * DM + c] + P[WPE][(r % T) * DM + c];
    for (int l = 0; l < L; l++) {
        LAct& a = A[l]; auto p = [&](int w) -> const D& { return P[pidx(l, w)]; };
        a.hin = h;
        a.n1 = norm_fwd_d(h, p(LN1G), p(LN1B), N, DM, LNEPS, false, a.m1, a.r1);
        a.q = lin(a.n1, p(WQ), p(BQ), DM, H * HD); a.k = lin(a.n1, p(WK), p(BK), DM, HKV * HD); a.v = lin(a.n1, p(WV), p(BV), DM, HKV * HD);
        a.qr = rope_d(a.q, H, false); a.kr = rope_d(a.k, HKV, false);
        attn_d(BT, T, H, HKV, HD, SCALE, a.qr, a.kr, a.v, a.att, a.lse, nullptr, nullptr, nullptr, nullptr);
        a.h1 = lin(a.att, p(WO), p(BO), H * HD, DM); for (int i = 0; i < N * DM; i++) a.h1[i] += h[i];
        a.n2 = norm_fwd_d(a.h1, p(LN2G), p(LN2B), N, DM, LNEPS, false, a.m2, a.r2);
        a.u = lin(a.n2, p(W1), p(B1_), DM, F); a.act.resize(a.u.size()); for (size_t i = 0; i < a.u.size(); i++) a.act[i] = gelu_d(a.u[i]);
        h = lin(a.act, p(W2), p(B2_), F, DM); for (int i = 0; i < N * DM; i++) h[i] += a.h1[i];
    }
    D mf, rf, nf = norm_fwd_d(h, P[LNFG], P[LNFB], N, DM, LNEPS, false, mf, rf);
    D logits = mm_d(nf, P[WTE], N, V, DM, false, true), dl;
    double loss = ce_d(logits, tgt, N, V, -100, G ? &dl : nullptr);
    if (!G) return loss;
    G->assign(NP, D()); for (int i = 0; i < NP; i++) (*G)[i].assign(P[i].size(), 0.0);
    auto addto = [](D& a, const D& b) { for (size_t i = 0; i < a.size(); i++) a[i] += b[i]; };
    (*G)[WTE] = mm_d(dl, nf, V, DM, N, true, false);
    D dnf = mm_d(dl, P[WTE], N, DM, V, false, false), dh, dg, db;
    norm_bwd_d(h, dnf, P[LNFG], N, DM, LNEPS, false, dh, (*G)[LNFG], (*G)[LNFB]);
    auto colsum = [](const D& x, int cols) { D s(cols, 0.0); for (size_t i = 0; i < x.size(); i++) s[i % cols] += x[i]; return s; };
    for (int l = L - 1; l >= 0; l--) {
        LAct& a = A[l]; auto p = [&](int w) -> const D& { return P[pidx(l, w)]; }; auto g = [&](int w) -> D& { return (*G)[pidx(l, w)]; };
        g(W2) = mm_d(a.act, dh, F, DM, N, true, false); g(B2_) = colsum(dh, DM);
        D dact = mm_d(dh, p(W2), N, F, DM, false, true), du(dact.size());
        for (size_t i = 0; i < du.size(); i++) du[i] = dact[i] * gelu_grad_d(a.u[i]);
        g(W1) = mm_d(a.n2, du, DM, F, N, true, false); g(B1_) = colsum(du, F);
        D dn2 = mm_d(du, p(W1), N, DM, F, false, true), dx;
        norm_bwd_d(a.h1, dn2, p(LN2G), N, DM, LNEPS, false, dx, g(LN2G), g(LN2B));
        D dh1 = dh; addto(dh1, dx);
        g(WO) = mm_d(a.att, dh1, H * HD, DM, N, true, false); g(BO) = colsum(dh1, DM);
        D da = mm_d(dh1, p(WO), N, H * HD, DM, false, true), o2, l2, dqr, dkr, dv;
        attn_d(BT, T, H, HKV, HD, SCALE, a.qr, a.kr, a.v, o2, l2, &da, &dqr, &dkr, &dv);
        D dq = rope_d(dqr, H, true), dk = rope_d(dkr, HKV, true);
        g(WQ) = mm_d(a.n1, dq, DM, H * HD, N, true, false); g(BQ) = colsum(dq, H * HD);
        g(WK) = mm_d(a.n1, dk, DM, HKV * HD, N, true, false); g(BK) = colsum(dk, HKV * HD);
        g(WV) = mm_d(a.n1, dv, DM, HKV * HD, N, true, false); g(BV) = colsum(dv, HKV * HD);
        D dn1 = mm_d(dq, p(WQ), N, DM, H * HD, false, true);
        addto(dn1, mm_d(dk, p(WK), N, DM, HKV * HD, false, true)); addto(dn1, mm_d(dv, p(WV), N, DM, HKV * HD, false, true));
        norm_bwd_d(a.hin, dn1, p(LN1G), N, DM, LNEPS, false, dx, g(LN1G), g(LN1B));
        dh = dh1; addto(dh, dx);
    }
    for (int r = 0; r < N; r++) for (int c = 0; c < DM; c++) { (*G)[WTE][ids[r] * DM + c] += dh[r * DM + c]; (*G)[WPE][(r % T) * DM + c] += dh[r * DM + c]; }
    return loss;
}

// ---------------------------------------------------------------------------------------------------------------------
// GPU model as a sequence of op launches
struct Gpu {
    Dev dv;
    std::vector<int> p, g, m, v, R, C;
    int ids, tgt, loss, rowloss, lse_f, logits, dlogits, step, part, nrm;
    int h0, nf, mf, rf, dnf, dh, dhn, tmp, dtmp;
    struct Lay { int n1, m1, r1, q, k, v, qr, kr, att, lse, h1, n2, m2, r2, u, act, h2, dact, du, dn2, dh1, da, delta, dqr, dkr, dvv, dq, dk, dn1; } ly[L];
    int nslots = 0;
    void mm(int a, int b, int out, int M, int Nn, int K, int fl, int bias = -1, int res = -1, int pre = -1) {
        Batch bt;
        for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < Nn; n0 += 64)
            bt.add(OP_matmul_tile, A().b(a).b(b).b(out).b(bias).b(res).b(pre).i(m0).i(n0).i(M).i(Nn).i(K).i(fl | 16).f(1.f));
        chk(run(dv, bt), "matmul_tile");
    }
    void chk(int e, const char* what) { if (e) { printf("device error %d in %s\n", e, what); exit(2); } }
    void one(int op, const A& a) { chk(run1(dv, op, a, 64 * 1024), "op"); }
    void rows(int op, std::function<A(int, int)> mk) {  // row-block tasks of 16 rows
        Batch bt; for (int r0 = 0; r0 < N; r0 += 16) bt.add(op, mk(r0, r0 + 16)); chk(run(dv, bt, 64 * 1024), "rows");
    }
    void ln_fwd(int x, int g_, int b_, int o, int mean, int rstd) { rows(OP_layernorm_fwd, [&](int a, int b) { return A().b(x).b(g_).b(b_).b(o).b(mean).b(rstd).i(a).i(b).i(DM).f((float)LNEPS); }); }
    void ln_bwd(int x, int dy, int gi, int mean, int rstd, int dx, int dg, int db, int acc) {
        rows(OP_layernorm_bwd_dx, [&](int a, int b) { return A().b(x).b(dy).b(gi).b(mean).b(rstd).b(dx).i(a).i(b).i(DM).i(acc); });
        one(OP_norm_bwd_params, A().b(x).b(dy).b(mean).b(rstd).b(dg).b(db).i(N).i(DM).i(0).i(DM).i(0));
    }
    void rope(int op, int x, int o, int heads) { rows(op, [&](int a, int b) { return A().b(x).b(o).b(-1).i(a).i(b).i(T).i(heads).i(HD).f((float)THETA); }); }
    Gpu() {
        shapes(R, C);
        for (int i = 0; i < NP; i++) { p.push_back(dv.addf(R[i], C[i])); g.push_back(dv.addf(R[i], C[i])); m.push_back(dv.addf(R[i], C[i])); v.push_back(dv.addf(R[i], C[i])); }
        ids = dv.addi(N); tgt = dv.addi(N); loss = dv.addf(1, 2); rowloss = dv.addf(1, N); lse_f = dv.addf(1, N);
        logits = dv.addf(N, V); dlogits = dv.addf(N, V); step = dv.addi(1); nslots = NP; part = dv.addf(1, nslots); nrm = dv.addf(1, 2);
        h0 = dv.addf(N, DM); nf = dv.addf(N, DM); mf = dv.addf(1, N); rf = dv.addf(1, N); dnf = dv.addf(N, DM); dh = dv.addf(N, DM); dhn = dv.addf(N, DM);
        tmp = dv.addf(N, DM); dtmp = dv.addf(N, DM);
        for (int l = 0; l < L; l++) {
            auto& y = ly[l]; auto f = [&](int c) { return dv.addf(N, c); }; auto s = [&]() { return dv.addf(1, N); };
            y.n1 = f(DM); y.m1 = s(); y.r1 = s(); y.q = f(H * HD); y.k = f(HKV * HD); y.v = f(HKV * HD); y.qr = f(H * HD); y.kr = f(HKV * HD);
            y.att = f(H * HD); y.lse = dv.addf(1, BT * H * T); y.h1 = f(DM); y.n2 = f(DM); y.m2 = s(); y.r2 = s(); y.u = f(F); y.act = f(F); y.h2 = f(DM);
            y.dact = f(F); y.du = f(F); y.dn2 = f(DM); y.dh1 = f(DM); y.da = f(H * HD); y.delta = dv.addf(1, BT * H * T); y.dqr = f(H * HD); y.dkr = f(HKV * HD);
            y.dvv = f(HKV * HD); y.dq = f(H * HD); y.dk = f(HKV * HD); y.dn1 = f(DM);
        }
    }
    int hin(int l) { return l == 0 ? h0 : ly[l - 1].h2; }
    void forward() {
        rows(OP_embedding_fwd, [&](int a, int b) { return A().b(ids).b(p[WTE]).b(p[WPE]).b(h0).i(a).i(b).i(T).i(DM); });
        for (int l = 0; l < L; l++) {
            auto& y = ly[l]; auto P = [&](int w) { return p[pidx(l, w)]; };
            ln_fwd(hin(l), P(LN1G), P(LN1B), y.n1, y.m1, y.r1);
            mm(y.n1, P(WQ), y.q, N, H * HD, DM, 0, P(BQ)); mm(y.n1, P(WK), y.k, N, HKV * HD, DM, 0, P(BK)); mm(y.n1, P(WV), y.v, N, HKV * HD, DM, 0, P(BV));
            rope(OP_rope_fwd, y.q, y.qr, H); rope(OP_rope_fwd, y.k, y.kr, HKV);
            Batch b; for (int bb = 0; bb < BT; bb++) for (int h = 0; h < H; h++) b.add(OP_attn_fwd, A().b(y.qr).b(y.kr).b(y.v).b(y.att).b(y.lse).i(bb).i(h).i(0).i(T).i(H).i(HKV).i(HD).f((float)SCALE));
            chk(run(dv, b, 64 * 1024), "attn_fwd");
            mm(y.att, P(WO), y.h1, N, DM, H * HD, 0, P(BO), hin(l));
            ln_fwd(y.h1, P(LN2G), P(LN2B), y.n2, y.m2, y.r2);
            mm(y.n2, P(W1), y.act, N, F, DM, 8, P(B1_), -1, y.u);
            mm(y.act, P(W2), y.h2, N, DM, F, 0, P(B2_), y.h1);
        }
        ln_fwd(ly[L - 1].h2, p[LNFG], p[LNFB], nf, mf, rf);
        mm(nf, p[WTE], logits, N, V, DM, 2);
        rows(OP_softmax_ce_fwd, [&](int a, int b) { return A().b(logits).b(tgt).b(rowloss).b(lse_f).i(a).i(b).i(V).i(-100); });
        one(OP_softmax_ce_reduce, A().b(rowloss).b(tgt).b(loss).i(N).i(-100));
    }
    void backward() {
        rows(OP_softmax_ce_bwd, [&](int a, int b) { return A().b(logits).b(tgt).b(lse_f).b(loss).b(dlogits).i(a).i(b).i(V).i(-100).f(1.f); });
        mm(dlogits, nf, g[WTE], V, DM, N, 1);
        mm(dlogits, p[WTE], dnf, N, DM, V, 0);
        ln_bwd(ly[L - 1].h2, dnf, p[LNFG], mf, rf, dh, g[LNFG], g[LNFB], 0);
        for (int l = L - 1; l >= 0; l--) {
            auto& y = ly[l]; auto P = [&](int w) { return p[pidx(l, w)]; }; auto G = [&](int w) { return g[pidx(l, w)]; };
            mm(y.act, dh, G(W2), F, DM, N, 1); one(OP_colsum, A().b(dh).b(G(B2_)).i(N).i(0).i(DM).i(0));
            mm(dh, P(W2), y.dact, N, F, DM, 2);
            one(OP_gelu_bwd, A().b(y.u).b(y.dact).b(y.du).i(0).i(N * F).i(0));
            mm(y.n2, y.du, G(W1), DM, F, N, 1); one(OP_colsum, A().b(y.du).b(G(B1_)).i(N).i(0).i(F).i(0));
            mm(y.du, P(W1), y.dn2, N, DM, F, 2);
            ln_bwd(y.h1, y.dn2, P(LN2G), y.m2, y.r2, dtmp, G(LN2G), G(LN2B), 0);
            one(OP_residual_add, A().b(dh).b(dtmp).b(y.dh1).i(0).i(N * DM));
            mm(y.att, y.dh1, G(WO), H * HD, DM, N, 1); one(OP_colsum, A().b(y.dh1).b(G(BO)).i(N).i(0).i(DM).i(0));
            mm(y.dh1, P(WO), y.da, N, H * HD, DM, 2);
            rows(OP_attn_bwd_delta, [&](int a, int b) { return A().b(y.att).b(y.da).b(y.delta).i(a).i(b).i(T).i(H).i(HD); });
            Batch b;
            for (int bb = 0; bb < BT; bb++) {
                for (int h = 0; h < H; h++) b.add(OP_attn_bwd_dq, A().b(y.qr).b(y.kr).b(y.v).b(y.da).b(y.lse).b(y.delta).b(y.dqr).i(bb).i(h).i(0).i(T).i(H).i(HKV).i(HD).f((float)SCALE));
                for (int h = 0; h < HKV; h++) b.add(OP_attn_bwd_dkv, A().b(y.qr).b(y.kr).b(y.v).b(y.da).b(y.lse).b(y.delta).b(y.dkr).b(y.dvv).i(bb).i(h).i(0).i(T).i(H).i(HKV).i(HD).f((float)SCALE));
            }
            chk(run(dv, b, 64 * 1024), "attn_bwd");
            rope(OP_rope_bwd, y.dqr, y.dq, H); rope(OP_rope_bwd, y.dkr, y.dk, HKV);
            mm(y.n1, y.dq, G(WQ), DM, H * HD, N, 1); one(OP_colsum, A().b(y.dq).b(G(BQ)).i(N).i(0).i(H * HD).i(0));
            mm(y.n1, y.dk, G(WK), DM, HKV * HD, N, 1); one(OP_colsum, A().b(y.dk).b(G(BK)).i(N).i(0).i(HKV * HD).i(0));
            mm(y.n1, y.dvv, G(WV), DM, HKV * HD, N, 1); one(OP_colsum, A().b(y.dvv).b(G(BV)).i(N).i(0).i(HKV * HD).i(0));
            mm(y.dq, P(WQ), y.dn1, N, DM, H * HD, 2); mm(y.dk, P(WK), y.dn1, N, DM, HKV * HD, 2 | 4); mm(y.dvv, P(WV), y.dn1, N, DM, HKV * HD, 2 | 4);
            ln_bwd(hin(l), y.dn1, P(LN1G), y.m1, y.r1, dtmp, G(LN1G), G(LN1B), 0);
            one(OP_residual_add, A().b(y.dh1).b(dtmp).b(dhn).i(0).i(N * DM));
            std::swap(dh, dhn);
        }
        for (int v0 = 0; v0 < V; v0 += 16) one(OP_embedding_bwd, A().b(ids).b(dh).b(g[WTE]).i(N).i(0).i(v0).i(v0 + 16).i(DM).i(1));
        one(OP_embedding_bwd, A().b(-1).b(dh).b(g[WPE]).i(N).i(T).i(0).i(T).i(DM).i(0));
    }
    void update() {
        Batch b;
        for (int i = 0; i < NP; i++) b.add(OP_gradnorm_partial, A().b(g[i]).b(part).i(i).i(0).i(R[i] * C[i]));
        chk(run(dv, b), "gradnorm_partial");
        one(OP_gradnorm_finish, A().b(part).i(NP).b(nrm).f((float)CLIP));
        one(OP_counter_inc, A().b(step).i(0).i(1).i(1));
        Batch ba;
        for (int i = 0; i < NP; i++) ba.add(OP_adamw, A().b(p[i]).b(g[i]).b(m[i]).b(v[i]).b(nrm).b(step).i(0).i(R[i] * C[i]).f((float)LR).f((float)B1).f((float)B2).f((float)EPS).f((float)WD));
        chk(run(dv, ba), "adamw");
    }
};

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("device: %s, %d SMs\n", pr.name, pr.multiProcessorCount);
    std::vector<int> R, C; shapes(R, C);
    Rng r(2024);
    std::vector<std::vector<float>> hp(NP);
    for (int i = 0; i < NP; i++) {
        bool gain = (i == LNFG) || (i >= LAYER0 && ((i - LAYER0) % PER == LN1G || (i - LAYER0) % PER == LN2G));
        bool bias = (i == LNFB) || (i >= LAYER0 && ((i - LAYER0) % PER == LN1B || (i - LAYER0) % PER == LN2B || (i - LAYER0) % PER == BQ || (i - LAYER0) % PER == BK ||
                                                   (i - LAYER0) % PER == BV || (i - LAYER0) % PER == BO || (i - LAYER0) % PER == B1_ || (i - LAYER0) % PER == B2_));
        hp[i] = rnd(r, R[i] * C[i], gain || bias ? 0.05 : (i == WTE || i == WPE ? 0.3 : 0.25));
        if (gain) for (auto& x : hp[i]) x += 1.f;
    }
    // data: next token = (5*x+3) mod 64 with prob 0.85, else random; last target of each row ignored
    auto make_batch = [&](Rng& rr, std::vector<int>& ids, std::vector<int>& tg) {
        ids.assign(N, 0); tg.assign(N, -100);
        for (int b = 0; b < BT; b++) {
            int x = rr.u32() % V;
            for (int t = 0; t < T; t++) {
                ids[b * T + t] = x;
                int nx = rr.uni() < 0.85 ? (5 * x + 3) % V : (int)(rr.u32() % V);
                if (t + 1 < T) tg[b * T + t] = nx;
                x = nx;
            }
        }
    };
    // ---- (1) CPU reference gradients vs finite differences
    std::vector<D> P(NP); for (int i = 0; i < NP; i++) P[i] = dbl(hp[i]);
    std::vector<int> ids, tg; { Rng rr(99); make_batch(rr, ids, tg); }
    std::vector<D> G0; double loss0 = cpu_model(P, ids, tg, &G0);
    {
        double me = 0, h = 1e-6; Rng rs(5); int cnt = 0;
        for (int i = 0; i < NP; i++) for (int s = 0; s < 6; s++) {
            size_t j = rs.u32() % P[i].size(); double sv = P[i][j];
            P[i][j] = sv + h; double lp = cpu_model(P, ids, tg, nullptr); P[i][j] = sv - h; double lm = cpu_model(P, ids, tg, nullptr); P[i][j] = sv;
            me = std::max(me, fabs((lp - lm) / (2 * h) - G0[i][j])); cnt++;
        }
        char d[96]; snprintf(d, sizeof d, "%d coordinates, max abs diff %.2e", cnt, me);
        verdict("e2e CPU double analytic gradients vs finite differences (all 36 tensors)", me < 1e-7, d);
    }
    // ---- (2) GPU gradients vs CPU at step 0
    Gpu gpu;
    for (int i = 0; i < NP; i++) gpu.dv.up(gpu.p[i], hp[i]);
    gpu.dv.upi(gpu.ids, std::vector<int32_t>(ids.begin(), ids.end())); gpu.dv.upi(gpu.tgt, std::vector<int32_t>(tg.begin(), tg.end()));
    gpu.forward(); gpu.backward();
    check("e2e step 0 loss GPU vs CPU", {gpu.dv.down(gpu.loss)[0]}, {loss0}, 1e-5);
    double worst = 0; int wi = 0;
    for (int i = 0; i < NP; i++) {
        double e = relerr(gpu.dv.down(gpu.g[i]), G0[i], 1e-3); if (e > worst) { worst = e; wi = i; }
    }
    char dd[96]; snprintf(dd, sizeof dd, "worst relerr %.2e (tensor %d), floor 1e-3", worst, wi);
    verdict("e2e step 0 gradients, all 36 tensors, GPU(3xTF32+fp32) vs CPU double", worst < 2e-4, dd);
    // ---- (3) trajectory: reset GPU state, same data stream
    for (int i = 0; i < NP; i++) { gpu.dv.up(gpu.p[i], hp[i]); std::vector<float> z(R[i] * C[i], 0.f); gpu.dv.up(gpu.m[i], z); gpu.dv.up(gpu.v[i], z); }
    gpu.dv.upi(gpu.step, {0});
    for (int i = 0; i < NP; i++) P[i] = dbl(hp[i]);
    std::vector<D> M(NP), Vv(NP); for (int i = 0; i < NP; i++) { M[i].assign(P[i].size(), 0.0); Vv[i].assign(P[i].size(), 0.0); }
    Rng rr(777); double maxd = 0, first = 0, last = 0; std::vector<float> gl, cl;
    for (int s = 0; s < STEPS; s++) {
        make_batch(rr, ids, tg);
        gpu.dv.upi(gpu.ids, std::vector<int32_t>(ids.begin(), ids.end())); gpu.dv.upi(gpu.tgt, std::vector<int32_t>(tg.begin(), tg.end()));
        gpu.forward(); gpu.backward();
        double lg = gpu.dv.down(gpu.loss)[0];
        std::vector<D> G; double lc = cpu_model(P, ids, tg, &G);
        gl.push_back((float)lg); cl.push_back((float)lc); maxd = std::max(maxd, fabs(lg - lc));
        if (s == 0) first = lc; last = lc;
        gpu.update();
        double sq = 0; for (auto& gg : G) for (double x : gg) sq += x * x;
        double nrm = sqrt(sq), coef = std::min(1.0, CLIP / (nrm + 1e-6));
        for (int i = 0; i < NP; i++) adamw_d(P[i], G[i], M[i], Vv[i], (float)LR, (float)B1, (float)B2, (float)EPS, (float)WD, s + 1, coef);
        if (s % 40 == 0 || s == STEPS - 1) printf("  step %3d  loss gpu %.6f cpu %.6f  |grad| %.3f clip %.3f\n", s, lg, lc, nrm, coef);
    }
    char d3[128]; snprintf(d3, sizeof d3, "max |loss_gpu - loss_cpu| over %d steps = %.3e (tol 2e-3)", STEPS, maxd);
    verdict("e2e loss trajectory GPU ops vs CPU double, step for step", maxd < 2e-3, d3);
    snprintf(d3, sizeof d3, "loss %.4f -> %.4f", first, last);
    verdict("e2e loss falls (final < 0.5 * initial)", last < 0.5 * first, d3);
    // parameters after 200 steps
    double pe = 0; for (int i = 0; i < NP; i++) { auto gp = gpu.dv.down(gpu.p[i]); for (size_t j = 0; j < gp.size(); j++) pe = std::max(pe, fabs(gp[j] - P[i][j])); }
    snprintf(d3, sizeof d3, "max abs param diff after %d steps = %.3e (tol 5e-3)", STEPS, pe);
    verdict("e2e parameters after training GPU vs CPU", pe < 5e-3, d3);
    // ---- (5) bitwise reproducibility: rerun the whole GPU trajectory
    {
        Gpu g2;
        for (int i = 0; i < NP; i++) g2.dv.up(g2.p[i], hp[i]);
        g2.dv.upi(g2.step, {0});
        Rng r2(777); for (int s = 0; s < STEPS; s++) {
            make_batch(r2, ids, tg);
            g2.dv.upi(g2.ids, std::vector<int32_t>(ids.begin(), ids.end())); g2.dv.upi(g2.tgt, std::vector<int32_t>(tg.begin(), tg.end()));
            g2.forward(); g2.backward(); g2.update();
        }
        bool same = true; for (int i = 0; i < NP; i++) same &= g2.dv.down(g2.p[i]) == gpu.dv.down(gpu.p[i]);
        verdict("e2e two independent 200-step GPU runs are bitwise identical", same, "");
    }
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
