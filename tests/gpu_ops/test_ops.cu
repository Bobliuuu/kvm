// Per-op unit tests for the GPT operator library against CPU double references, plus edge cases and error codes.
// Build: nvcc -O2 -arch=sm_80 -std=c++17 -I../../include -I. test_ops.cu -o test_ops    Run: ./test_ops
#include <algorithm>
#include <map>
#include "harness.cuh"
#include "refs.hpp"


// ---------------------------------------------------------------------------------------------------------------------
static void t_philox() {
    uint32_t o[4], c0[4] = {0, 0, 0, 0}, c1[4] = {~0u, ~0u, ~0u, ~0u};
    philox_host(c0, 0, 0, o);
    bool a = o[0] == 0x6627e8d5 && o[1] == 0xe169c58d && o[2] == 0xbc57ac4c && o[3] == 0x9b00dbd8;
    philox_host(c1, ~0u, ~0u, o);
    bool b = o[0] == 0x408f276d && o[1] == 0x41c83b0e && o[2] == 0xa20bc7c6 && o[3] == 0x6d5451fd;
    check_true("philox4x32-10 known-answer vectors (host)", a && b);
}

static void t_embedding() {
    Rng r(11);
    for (int variant = 0; variant < 2; variant++) {
        int V = 50, d = variant ? 1 : 20, N = 37, T = 16;
        Dev dv;
        int ids = dv.addi(N), wte = dv.addf(V, d, d + 3), wpe = dv.addf(T, d), out = dv.addf(N, d, d + 1);
        std::vector<int32_t> hid(N); for (auto& x : hid) x = r.u32() % V;
        hid[3] = hid[4] = hid[5] = 7;  // duplicates
        auto hw = rnd(r, V * d), hp = rnd(r, T * d);
        dv.upi(ids, hid); dv.up(wte, hw); dv.up(wpe, hp);
        Batch b;
        for (int r0 = 0; r0 < N; r0 += 10) b.add(OP_embedding_fwd, A().b(ids).b(wte).b(wpe).b(out).i(r0).i(std::min(N, r0 + 10)).i(T).i(d));
        int e = run(dv, b);
        D ref(N * d); for (int i = 0; i < N; i++) for (int c = 0; c < d; c++) ref[i * d + c] = (double)hw[hid[i] * d + c] + hp[(i % T) * d + c];
        check("embedding_fwd d=" + std::to_string(d), dv.down(out), ref, 1e-7);
        check_true("embedding_fwd err==0", e == 0);
        // backward: token table by key range tasks, positional table with ids=-1
        auto hdo = rnd(r, N * d);
        int dout = dv.addf(N, d), dwte = dv.addf(V, d, d + 3), dwpe = dv.addf(T, d);
        dv.up(dout, hdo); dv.up(dwte, rnd(r, V * d)); dv.up(dwpe, rnd(r, T * d));
        for (int acc = 0; acc < 2; acc++) {
            auto w0 = dv.down(dwte), p0 = dv.down(dwpe);
            Batch b2;
            for (int v0 = 0; v0 < V; v0 += 16) b2.add(OP_embedding_bwd, A().b(ids).b(dout).b(dwte).i(N).i(0).i(v0).i(std::min(V, v0 + 16)).i(d).i(acc));
            b2.add(OP_embedding_bwd, A().b(-1).b(dout).b(dwpe).i(N).i(T).i(0).i(T).i(d).i(acc));
            e = run(dv, b2);
            D rw(V * d, 0.0), rp(T * d, 0.0);
            if (acc) { for (int i = 0; i < V * d; i++) rw[i] = w0[i]; for (int i = 0; i < T * d; i++) rp[i] = p0[i]; }
            for (int i = 0; i < N; i++) for (int c = 0; c < d; c++) { rw[hid[i] * d + c] += hdo[i * d + c]; rp[(i % T) * d + c] += hdo[i * d + c]; }
            check("embedding_bwd tok d=" + std::to_string(d) + " acc=" + std::to_string(acc), dv.down(dwte), rw, 1e-6);
            check("embedding_bwd pos d=" + std::to_string(d) + " acc=" + std::to_string(acc), dv.down(dwpe), rp, 1e-6);
            if (!acc) { auto g1 = dv.down(dwte); run(dv, b2); check_true("embedding_bwd bitwise deterministic", g1 == dv.down(dwte)); }
        }
        if (variant == 0) {   // out-of-range token id -> error 5
            hid[2] = V + 5; dv.upi(ids, hid);
            check_true("embedding_fwd bad token -> err 5", run1(dv, OP_embedding_fwd, A().b(ids).b(wte).b(wpe).b(out).i(0).i(N).i(T).i(d)) == KVM_GPT_E_TOKEN);
        }
    }
}

static void t_matmul() {
    Rng r(21);
    struct S { int M, N, K; } shapes[] = {{1, 1, 1}, {5, 7, 3}, {64, 64, 32}, {65, 63, 33}, {130, 70, 100}, {1, 300, 17}, {70, 1, 90}};
    for (int prec = 0; prec < 2; prec++) for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) for (auto s : shapes) for (int full = 0; full < 2; full++) {
        int M = s.M, N = s.N, K = s.K;
        Dev dv;
        int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
        int ia = dv.addf(ar, ac, ac + ((M + N) & 1 ? 1 : 4)), ib = dv.addf(br, bc, bc + 4), io = dv.addf(M, N, N + 1);
        int ibias = dv.addf(1, N), ires = dv.addf(M, N), ipre = dv.addf(M, N, N + 4);
        auto ha = rnd(r, ar * ac), hb = rnd(r, br * bc), ho = rnd(r, M * N), hbias = rnd(r, N), hres = rnd(r, M * N);
        dv.up(ia, ha); dv.up(ib, hb); dv.up(io, ho); dv.up(ibias, hbias); dv.up(ires, hres);
        float alpha = full ? 0.7f : 1.f;
        int fl = ta | (tb << 1) | (prec << 4) | (full ? (4 | 8) : 0);
        Batch b;
        for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64)
            b.add(OP_matmul_tile, A().b(ia).b(ib).b(io).b(full ? ibias : -1).b(full ? ires : -1).b(full ? ipre : -1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
        int e = run(dv, b);
        D C = mm_d(dbl(ha), dbl(hb), M, N, K, ta, tb), ref(M * N), pre(M * N);
        for (int i = 0; i < M * N; i++) {
            double v = alpha * C[i];
            if (full) { v += hbias[i % N]; pre[i] = v; v = gelu_d(v) + hres[i]; v += ho[i]; }
            ref[i] = v;
        }
        char nm[128]; snprintf(nm, sizeof nm, "matmul_tile %s ta=%d tb=%d %dx%dx%d %s", prec ? "3xTF32" : "TF32", ta, tb, M, N, K, full ? "bias+gelu+res+acc" : "plain");
        double tol = prec ? 2e-5 : 6e-3;
        check(nm, dv.down(io), ref, tol);
        if (full) check(std::string(nm) + " pre", dv.down(ipre), pre, tol);
        if (e) check_true(std::string(nm) + " err", false);
    }
    {   // in-place residual: res aliases out, accumulate off
        int M = 33, N = 40, K = 20; Dev dv;
        int ia = dv.addf(M, K), ib = dv.addf(K, N), io = dv.addf(M, N);
        auto ha = rnd(r, M * K), hb = rnd(r, K * N), ho = rnd(r, M * N);
        dv.up(ia, ha); dv.up(ib, hb); dv.up(io, ho);
        run1(dv, OP_matmul_tile, A().b(ia).b(ib).b(io).b(-1).b(io).b(-1).i(0).i(0).i(M).i(N).i(K).i(16).f(1.f));
        D C = mm_d(dbl(ha), dbl(hb), M, N, K, 0, 0); for (int i = 0; i < M * N; i++) C[i] += ho[i];
        check("matmul_tile res aliasing out", dv.down(io), C, 2e-5);
    }
}

static void norm_case(bool rms, int R, int d) {
    Rng r(31 + d);
    Dev dv;
    int x = dv.addf(R, d, d + 2), g = dv.addf(1, d), be = dv.addf(1, d), o = dv.addf(R, d), mean = dv.addf(1, R), rstd = dv.addf(1, R);
    int dy = dv.addf(R, d), dx = dv.addf(R, d, d + 1), dg = dv.addf(1, d), db = dv.addf(1, d);
    auto hx = rnd(r, R * d, 2.0), hg = rnd(r, d), hb = rnd(r, d), hdy = rnd(r, R * d);
    for (auto& v : hx) v += 0.5f;
    dv.up(x, hx); dv.up(g, hg); dv.up(be, hb); dv.up(dy, hdy);
    const float eps = 1e-5f;
    std::string nm = std::string(rms ? "rmsnorm" : "layernorm") + " R=" + std::to_string(R) + " d=" + std::to_string(d);
    Batch b;
    for (int r0 = 0; r0 < R; r0 += 5) {
        if (rms) b.add(OP_rmsnorm_fwd, A().b(x).b(g).b(o).b(rstd).i(r0).i(std::min(R, r0 + 5)).i(d).f(eps));
        else b.add(OP_layernorm_fwd, A().b(x).b(g).b(be).b(o).b(mean).b(rstd).i(r0).i(std::min(R, r0 + 5)).i(d).f(eps));
    }
    int e = run(dv, b);
    D rm, rr, ro = norm_fwd_d(dbl(hx), dbl(hg), rms ? D() : dbl(hb), R, d, eps, rms, rm, rr);
    check(nm + " fwd out", dv.down(o), ro, 1e-5);
    check(nm + " fwd rstd", dv.down(rstd), rr, 1e-5);
    D rdx, rdg, rdb; norm_bwd_d(dbl(hx), dbl(hdy), dbl(hg), R, d, eps, rms, rdx, rdg, rdb);
    for (int acc = 0; acc < 2; acc++) {
        dv.up(dx, rnd(r, R * d)); dv.up(dg, rnd(r, d)); dv.up(db, rnd(r, d));
        auto dx0 = dv.down(dx), dg0 = dv.down(dg), db0 = dv.down(db);
        Batch b2;
        for (int r0 = 0; r0 < R; r0 += 5) {
            if (rms) b2.add(OP_rmsnorm_bwd_dx, A().b(x).b(dy).b(g).b(rstd).b(dx).i(r0).i(std::min(R, r0 + 5)).i(d).i(acc));
            else b2.add(OP_layernorm_bwd_dx, A().b(x).b(dy).b(g).b(mean).b(rstd).b(dx).i(r0).i(std::min(R, r0 + 5)).i(d).i(acc));
        }
        for (int c0 = 0; c0 < d; c0 += 100) b2.add(OP_norm_bwd_params, A().b(x).b(dy).b(rms ? -1 : mean).b(rstd).b(dg).b(rms ? -1 : db).i(R).i(d).i(c0).i(std::min(d, c0 + 100)).i(acc));
        e |= run(dv, b2);
        D e1 = rdx, e2 = rdg, e3 = rdb;
        if (acc) { for (int i = 0; i < R * d; i++) e1[i] += dx0[i]; for (int i = 0; i < d; i++) { e2[i] += dg0[i]; e3[i] += db0[i]; } }
        std::string s = nm + " bwd acc=" + std::to_string(acc);
        check(s + " dx", dv.down(dx), e1, 2e-5, d == 1 ? 1.0 : 1e-30);
        check(s + " dgamma", dv.down(dg), e2, 2e-5);
        if (!rms) check(s + " dbeta", dv.down(db), e3, 2e-5);
    }
    if (e) check_true(nm + " err", false);
}
static void t_norms() { for (int rms = 0; rms < 2; rms++) for (int R : {1, 13}) for (int d : {1, 7, 32, 100, 1000}) norm_case(rms, R, d); }

static void t_elementwise() {
    Rng r(41);
    int n = 1000; Dev dv;
    int x = dv.addf(1, n), y = dv.addf(1, n), o = dv.addf(1, n), dy = dv.addf(1, n), dx = dv.addf(1, n);
    auto hx = rnd(r, n, 3.0), hy = rnd(r, n), hdy = rnd(r, n);
    dv.up(x, hx); dv.up(y, hy); dv.up(dy, hdy);
    Batch b; for (int e0 = 0; e0 < n; e0 += 300) b.add(OP_gelu_fwd, A().b(x).b(o).i(e0).i(std::min(n, e0 + 300)));
    run(dv, b); D ref(n); for (int i = 0; i < n; i++) ref[i] = gelu_d(hx[i]);
    check("gelu_fwd", dv.down(o), ref, 1e-6);
    Batch b2; for (int e0 = 0; e0 < n; e0 += 300) b2.add(OP_gelu_bwd, A().b(x).b(dy).b(dx).i(e0).i(std::min(n, e0 + 300)).i(0));
    run(dv, b2); for (int i = 0; i < n; i++) ref[i] = hdy[i] * gelu_grad_d(hx[i]);
    check("gelu_bwd", dv.down(dx), ref, 1e-6);
    // gelu derivative vs central finite difference of gelu_d (reference self-check)
    double md = 0; for (int i = 0; i < 50; i++) { double h = 1e-6, fd = (gelu_d(hx[i] + h) - gelu_d(hx[i] - h)) / (2 * h); md = std::max(md, fabs(fd - gelu_grad_d(hx[i]))); }
    check_true("gelu_grad reference vs finite difference", md < 1e-7);
    run1(dv, OP_residual_add, A().b(x).b(y).b(x).i(0).i(n));  // in place
    for (int i = 0; i < n; i++) ref[i] = (double)hx[i] + hy[i];
    check("residual_add (out aliases a)", dv.down(x), ref, 1e-7);
    // zero_buf / counter_inc / colsum
    run1(dv, OP_zero_buf, A().b(y).i(100).i(900));
    auto hy2 = dv.down(y); bool okz = true; for (int i = 0; i < n; i++) okz &= (i >= 100 && i < 900) ? hy2[i] == 0.f : hy2[i] == hy[i];
    check_true("zero_buf range", okz);
    Dev d2; int ctr = d2.addi(8); d2.upi(ctr, {1, 2, 3, 4, 5, 6, 7, 8});
    run1(d2, OP_counter_inc, A().b(ctr).i(2).i(5).i(10));
    check_true("counter_inc range", d2.downi(ctr) == std::vector<int32_t>({1, 2, 13, 14, 15, 6, 7, 8}));
    int R = 37, C = 300; Dev d3; int cx = d3.addf(R, C, C + 3), co = d3.addf(1, C);
    auto hc = rnd(r, R * C), ho0 = rnd(r, C); d3.up(cx, hc); d3.up(co, ho0);
    Batch b3; for (int c0 = 0; c0 < C; c0 += 128) b3.add(OP_colsum, A().b(cx).b(co).i(R).i(c0).i(std::min(C, c0 + 128)).i(1));
    run(d3, b3); D cs(C, 0); for (int c = 0; c < C; c++) { cs[c] = ho0[c]; for (int i = 0; i < R; i++) cs[c] += hc[i * C + c]; }
    check("colsum acc", d3.down(co), cs, 1e-6);
}

static void t_rope() {
    Rng r(51);
    int R = 24, T = 8, H = 3, hd = 16; Dev dv;
    int x = dv.addf(R, H * hd), o = dv.addf(R, H * hd), o2 = dv.addf(R, H * hd), pos = dv.addi(1);
    auto hx = rnd(r, R * H * hd); dv.up(x, hx); dv.upi(pos, {5});
    for (int use_pos = 0; use_pos < 2; use_pos++) {
        Batch b; for (int r0 = 0; r0 < R; r0 += 7) b.add(OP_rope_fwd, A().b(x).b(o).b(use_pos ? pos : -1).i(r0).i(std::min(R, r0 + 7)).i(T).i(H).i(hd).f(10000.f));
        run(dv, b);
        D ref(R * H * hd);
        for (int i = 0; i < R; i++) for (int h = 0; h < H; h++) for (int j = 0; j < hd / 2; j++) {
            double ang = (i % T + (use_pos ? 5 : 0)) * pow(10000.0, -2.0 * j / hd), c = cos(ang), s = sin(ang);
            double x1 = hx[i * H * hd + h * hd + j], x2 = hx[i * H * hd + h * hd + j + hd / 2];
            ref[i * H * hd + h * hd + j] = x1 * c - x2 * s; ref[i * H * hd + h * hd + j + hd / 2] = x1 * s + x2 * c;
        }
        check(std::string("rope_fwd pos=") + (use_pos ? "buf" : "none"), dv.down(o), ref, 2e-6);
        Batch b2; b2.add(OP_rope_bwd, A().b(o).b(o2).b(use_pos ? pos : -1).i(0).i(R).i(T).i(H).i(hd).f(10000.f));
        run(dv, b2); check("rope_bwd inverts rope_fwd", dv.down(o2), dbl(hx), 2e-6);
    }
}

// ---- attention --------------------------------------------------------------------------------------------------------
struct AttnCfg { int B, T, H, Hkv, hd; };
static void attn_case(AttnCfg c) {
    Rng r(61 + c.hd + c.T);
    int B = c.B, T = c.T, H = c.H, Hkv = c.Hkv, hd = c.hd, R = B * T;
    double scale = 1.0 / sqrt((double)hd);
    int smem = kvm_gpt_smem_bytes(hd);
    char nm[96]; snprintf(nm, sizeof nm, "attn B=%d T=%d H=%d Hkv=%d hd=%d", B, T, H, Hkv, hd);
    Dev dv;
    int q = dv.addf(R, H * hd, H * hd + 4), k = dv.addf(R, Hkv * hd, Hkv * hd + 4), v = dv.addf(R, Hkv * hd), o = dv.addf(R, H * hd);
    int lse = dv.addf(1, B * H * T), dO = dv.addf(R, H * hd), delta = dv.addf(1, B * H * T), dq = dv.addf(R, H * hd, H * hd + 4);
    int dk = dv.addf(R, Hkv * hd, Hkv * hd + 4), dvv = dv.addf(R, Hkv * hd);
    auto hq = rnd(r, R * H * hd), hk = rnd(r, R * Hkv * hd), hv = rnd(r, R * Hkv * hd), hdo = rnd(r, R * H * hd);
    for (auto& x : hq) x *= 1.5f;
    dv.up(q, hq); dv.up(k, hk); dv.up(v, hv); dv.up(dO, hdo);
    Batch b;
    for (int bb = 0; bb < B; bb++) for (int h = 0; h < H; h++) for (int q0 = 0; q0 < T; q0 += 32)
        b.add(OP_attn_fwd, A().b(q).b(k).b(v).b(o).b(lse).i(bb).i(h).i(q0).i(T).i(H).i(Hkv).i(hd).f((float)scale));
    int e = run(dv, b, smem);
    D ro, rl, rdq, rdk, rdv;
    D Dq = dbl(hq), Dk = dbl(hk), Dv = dbl(hv), Ddo = dbl(hdo);
    attn_d(B, T, H, Hkv, hd, scale, Dq, Dk, Dv, ro, rl, &Ddo, &rdq, &rdk, &rdv);
    check(std::string(nm) + " fwd o", dv.down(o), ro, 2e-5);
    check(std::string(nm) + " fwd lse", dv.down(lse), rl, 2e-5);
    Batch b2; for (int r0 = 0; r0 < R; r0 += 40) b2.add(OP_attn_bwd_delta, A().b(o).b(dO).b(delta).i(r0).i(std::min(R, r0 + 40)).i(T).i(H).i(hd));
    e |= run(dv, b2, smem);
    Batch b3;
    for (int bb = 0; bb < B; bb++) {
        for (int h = 0; h < H; h++) for (int q0 = 0; q0 < T; q0 += 32)
            b3.add(OP_attn_bwd_dq, A().b(q).b(k).b(v).b(dO).b(lse).b(delta).b(dq).i(bb).i(h).i(q0).i(T).i(H).i(Hkv).i(hd).f((float)scale));
        for (int h = 0; h < Hkv; h++) for (int k0 = 0; k0 < T; k0 += 32)
            b3.add(OP_attn_bwd_dkv, A().b(q).b(k).b(v).b(dO).b(lse).b(delta).b(dk).b(dvv).i(bb).i(h).i(k0).i(T).i(H).i(hd).f((float)scale));
    }
    e |= run(dv, b3, smem);
    double fl = T == 1 ? 1e-2 : 1e-30;
    check(std::string(nm) + " bwd dq", dv.down(dq), rdq, 3e-5, fl);
    check(std::string(nm) + " bwd dk", dv.down(dk), rdk, 3e-5, fl);
    check(std::string(nm) + " bwd dv", dv.down(dvv), rdv, 3e-5);
    auto g1 = dv.down(dk); run(dv, b3, smem);
    check_true(std::string(nm) + " bwd bitwise deterministic", g1 == dv.down(dk));
    check_true(std::string(nm) + " err==0", e == 0);
}
static void t_attention() {
    for (AttnCfg c : {AttnCfg{2, 1, 2, 2, 16}, AttnCfg{1, 2, 2, 2, 16}, AttnCfg{2, 37, 4, 2, 32}, AttnCfg{1, 100, 4, 1, 64}, AttnCfg{1, 70, 2, 2, 128}, AttnCfg{2, 32, 2, 2, 16}, AttnCfg{1, 65, 4, 4, 32}, AttnCfg{2, 70, 16, 4, 64}})
        attn_case(c);
    {   // reference self-check: analytic CPU backward vs central finite differences of the CPU forward
        int B = 1, T = 5, H = 2, Hkv = 1, hd = 4; Rng r(7); double sc = 0.5;
        D q(B * T * H * hd), k(B * T * Hkv * hd), v(k.size()), dO(q.size()), o, lse, dq, dk, dv, o2, l2;
        for (auto* a : {&q, &k, &v, &dO}) for (auto& x : *a) x = r.nrm();
        attn_d(B, T, H, Hkv, hd, sc, q, k, v, o, lse, &dO, &dq, &dk, &dv);
        auto L = [&](const D& qq, const D& kk, const D& vv) { D oo, ll; attn_d(B, T, H, Hkv, hd, sc, qq, kk, vv, oo, ll, nullptr, nullptr, nullptr, nullptr); double s = 0; for (size_t i = 0; i < oo.size(); i++) s += oo[i] * dO[i]; return s; };
        double me = 0, h = 1e-6;
        for (size_t i = 0; i < q.size(); i++) { D a = q, b2 = q; a[i] += h; b2[i] -= h; me = std::max(me, fabs((L(a, k, v) - L(b2, k, v)) / (2 * h) - dq[i])); }
        for (size_t i = 0; i < k.size(); i++) { D a = k, b2 = k; a[i] += h; b2[i] -= h; me = std::max(me, fabs((L(q, a, v) - L(q, b2, v)) / (2 * h) - dk[i])); }
        for (size_t i = 0; i < v.size(); i++) { D a = v, b2 = v; a[i] += h; b2[i] -= h; me = std::max(me, fabs((L(q, k, a) - L(q, k, b2)) / (2 * h) - dv[i])); }
        char d[64]; snprintf(d, sizeof d, "max abs diff %.2e", me);
        verdict("attention reference backward vs finite differences (GQA)", me < 1e-7, d);
    }
}

// ---- serving ----------------------------------------------------------------------------------------------------------
static void t_decode() {
    Rng r(71);
    const int B = 3, H = 4, Hkv = 2, hd = 32, Tmax = 70; const double scale = 1.0 / sqrt((double)hd);
    std::vector<int> len = {0, 17, 69};
    Dev dv;
    int kn = dv.addf(B, Hkv * hd), vn = dv.addf(B, Hkv * hd), kc = dv.addf(B * Hkv * Tmax, hd), vc = dv.addf(B * Hkv * Tmax, hd);
    int lens = dv.addi(B), q = dv.addf(B, H * hd), o = dv.addf(B, H * hd);
    std::vector<std::vector<float>> K(B), V(B);   // per sequence tokens x (Hkv*hd)
    for (int b = 0; b < B; b++) { K[b] = rnd(r, (len[b] + 1) * Hkv * hd); V[b] = rnd(r, (len[b] + 1) * Hkv * hd); }
    auto hq = rnd(r, B * H * hd); dv.up(q, hq);
    for (int s = 0; s <= 69; s++) {   // append one token per active sequence per step (token s), then bump lens
        std::vector<float> hk(B * Hkv * hd, 0.f), hv(B * Hkv * hd, 0.f); Batch b, bi;
        for (int bb = 0; bb < B; bb++) if (s < len[bb] + 1) {
            for (int i = 0; i < Hkv * hd; i++) { hk[bb * Hkv * hd + i] = K[bb][s * Hkv * hd + i]; hv[bb * Hkv * hd + i] = V[bb][s * Hkv * hd + i]; }
            b.add(OP_kv_append, A().b(kn).b(vn).b(kc).b(vc).b(lens).i(bb).i(bb + 1).i(1).i(Hkv).i(hd).i(Tmax));
            if (s < len[bb]) bi.add(OP_counter_inc, A().b(lens).i(bb).i(bb + 1).i(1));
        }
        if (b.op.empty()) break;
        dv.up(kn, hk); dv.up(vn, hv);
        if (run(dv, b)) check_true("kv_append err", false);
        if (!bi.op.empty()) run(dv, bi);
    }
    check_true("lens after appends", dv.downi(lens) == std::vector<int32_t>({0, 17, 69}));
    Batch b; for (int bb = 0; bb < B; bb++) for (int h = 0; h < H; h++) b.add(OP_attn_decode, A().b(q).b(kc).b(vc).b(lens).b(o).i(bb).i(h).i(H).i(Hkv).i(hd).i(Tmax).f((float)scale));
    int e = run(dv, b);
    D ref(B * H * hd);
    for (int bb = 0; bb < B; bb++) for (int h = 0; h < H; h++) {
        int n = len[bb] + 1, kvh = h / (H / Hkv); D s(n); double mx = -1e300;
        for (int j = 0; j < n; j++) { double a = 0; for (int d = 0; d < hd; d++) a += (double)hq[bb * H * hd + h * hd + d] * K[bb][(j * Hkv + kvh) * hd + d]; s[j] = a * scale; mx = std::max(mx, s[j]); }
        double l = 0; for (auto& x : s) { x = exp(x - mx); l += x; }
        for (int d = 0; d < hd; d++) { double a = 0; for (int j = 0; j < n; j++) a += s[j] / l * V[bb][(j * Hkv + kvh) * hd + d]; ref[bb * H * hd + h * hd + d] = a; }
    }
    check("attn_decode (lens 0/17/69, GQA 2:1) vs reference", dv.down(o), ref, 2e-5);
    check_true("attn_decode err==0", e == 0);
    // prefill-style append (n_new>1) places tokens at lens[b]..
    Dev d2; int kn2 = d2.addf(2 * 5, hd), vn2 = d2.addf(2 * 5, hd), kc2 = d2.addf(2 * Tmax, hd), vc2 = d2.addf(2 * Tmax, hd), ln = d2.addi(2);
    auto hk2 = rnd(r, 10 * hd); d2.up(kn2, hk2); d2.up(vn2, hk2); d2.upi(ln, {3, 10});
    int e2 = run1(d2, OP_kv_append, A().b(kn2).b(vn2).b(kc2).b(vc2).b(ln).i(0).i(2).i(5).i(1).i(hd).i(Tmax));
    auto c2 = d2.down(kc2); bool ok = e2 == 0;
    for (int b2 = 0; b2 < 2; b2++) for (int t = 0; t < 5; t++) for (int c = 0; c < hd; c++) ok &= c2[(b2 * Tmax + (b2 ? 10 : 3) + t) * hd + c] == hk2[(b2 * 5 + t) * hd + c];
    check_true("kv_append n_new=5 at per-sequence offsets", ok);
    d2.upi(ln, {3, 66});
    check_true("kv_append overflow -> err 6", run1(d2, OP_kv_append, A().b(kn2).b(vn2).b(kc2).b(vc2).b(ln).i(0).i(2).i(5).i(1).i(hd).i(Tmax)) == KVM_GPT_E_KVLEN);
}

// ---- loss -------------------------------------------------------------------------------------------------------------
static void ce_case(int N, int V, double lscale, int mode /*0 none ignored,1 some,2 all*/, int ign, float gs) {
    Rng r(81 + N + V);
    Dev dv;
    int lg = dv.addf(N, V, V + 1), tg = dv.addi(N), rl = dv.addf(1, N), ls = dv.addf(1, N), loss = dv.addf(1, 2), dl = dv.addf(N, V, V + 2);
    auto hl = rnd(r, N * V, lscale); std::vector<int32_t> ht(N);
    for (int i = 0; i < N; i++) ht[i] = mode == 2 ? ign : ((mode == 1 && i % 2) ? ign : (int)(r.u32() % V));
    if (lscale >= 80 && N * V > 4) { hl[0] = 80.f; hl[1 % (N * V)] = -80.f; }
    dv.up(lg, hl); dv.upi(tg, ht);
    Batch b; for (int r0 = 0; r0 < N; r0 += 4) b.add(OP_softmax_ce_fwd, A().b(lg).b(tg).b(rl).b(ls).i(r0).i(std::min(N, r0 + 4)).i(V).i(ign));
    int e = run(dv, b);
    e |= run1(dv, OP_softmax_ce_reduce, A().b(rl).b(tg).b(loss).i(N).i(ign));
    Batch b2; for (int r0 = 0; r0 < N; r0 += 4) b2.add(OP_softmax_ce_bwd, A().b(lg).b(tg).b(ls).b(loss).b(dl).i(r0).i(std::min(N, r0 + 4)).i(V).i(ign).f(gs));
    e |= run(dv, b2);
    std::vector<int> it(ht.begin(), ht.end()); D rdl; double rloss = ce_d(dbl(hl), it, N, V, ign, &rdl);
    for (auto& x : rdl) x *= gs;
    char nm[96]; snprintf(nm, sizeof nm, "softmax_ce N=%d V=%d logit-scale=%g ignored=%s scale=%g", N, V, lscale, mode == 0 ? "none" : mode == 1 ? "half" : "all", gs);
    auto gl = dv.down(loss);
    check(std::string(nm) + " loss", {gl[0]}, {rloss}, 1e-5);
    check(std::string(nm) + " dlogits", dv.down(dl), rdl, 1e-5);
    check_true(std::string(nm) + " err==0", e == 0);
}
static void t_ce() {
    ce_case(13, 1000, 1.0, 0, -100, 1.f); ce_case(1, 1, 1.0, 0, -100, 1.f); ce_case(9, 7, 80.0, 0, -100, 1.f);
    ce_case(12, 50, 1.0, 1, -100, 0.5f); ce_case(6, 33, 1.0, 2, -1, 1.f); ce_case(5, 40000, 20.0, 0, -100, 1.f);
    // reference self-check: dlogits vs central finite differences of the mean loss
    int N = 4, V = 6; Rng r(5); D lg(N * V); for (auto& x : lg) x = r.nrm() * 3; std::vector<int> tg = {1, 2, -100, 5}; D dl; ce_d(lg, tg, N, V, -100, &dl);
    double me = 0, h = 1e-6; for (int i = 0; i < N * V; i++) { D a = lg, b = lg; a[i] += h; b[i] -= h; me = std::max(me, fabs((ce_d(a, tg, N, V, -100, nullptr) - ce_d(b, tg, N, V, -100, nullptr)) / (2 * h) - dl[i])); }
    check_true("softmax_ce reference gradient vs finite differences", me < 1e-8);
}

// ---- optimizer --------------------------------------------------------------------------------------------------------
static void t_adamw() {
    Rng r(91); int n = 1000; Dev dv;
    int p = dv.addf(1, n), g = dv.addf(1, n), m = dv.addf(1, n), v = dv.addf(1, n), gsb = dv.addf(1, 1), st = dv.addi(1), part = dv.addf(1, 4), nrm = dv.addf(1, 2);
    auto hp = rnd(r, n); D P = dbl(hp), M(n, 0.0), Vv(n, 0.0); dv.up(p, hp);
    dv.upi(st, {0}); dv.up(gsb, {0.5f});
    double lr = 1e-2, b1 = 0.9, b2 = 0.99, eps = 1e-8, wd = 0.1; bool ok = true; double me = 0;
    for (int t = 1; t <= 6; t++) {
        auto hg = rnd(r, n); dv.up(g, hg);
        run1(dv, OP_counter_inc, A().b(st).i(0).i(1).i(1));
        Batch b; for (int e0 = 0; e0 < n; e0 += 300) b.add(OP_adamw, A().b(p).b(g).b(m).b(v).b(gsb).b(st).i(e0).i(std::min(n, e0 + 300)).f((float)lr).f((float)b1).f((float)b2).f((float)eps).f((float)wd));
        ok &= run(dv, b) == 0;
        adamw_d(P, dbl(hg), M, Vv, (float)lr, (float)b1, (float)b2, (float)eps, (float)wd, t, 0.5);
        auto gp = dv.down(p); for (int i = 0; i < n; i++) me = std::max(me, fabs(gp[i] - P[i]));
    }
    char d[64]; snprintf(d, sizeof d, "max abs param diff %.2e after 6 steps", me);
    verdict("adamw (bias correction, decoupled wd, gscale) vs double reference", ok && me < 2e-6, d);
    check_true("adamw step=0 -> err 3", [&] { dv.upi(st, {0}); return run1(dv, OP_adamw, A().b(p).b(g).b(m).b(v).b(-1).b(st).i(0).i(10).f(1e-3f).f(.9f).f(.99f).f(1e-8f).f(0.f)) == KVM_GPT_E_RANGE; }());
    // grad norm / clip
    auto g1 = rnd(r, 700), g2 = rnd(r, 300); int ga = dv.addf(1, 700), gb = dv.addf(1, 300); dv.up(ga, g1); dv.up(gb, g2);
    Batch b; b.add(OP_gradnorm_partial, A().b(ga).b(part).i(0).i(0).i(350)); b.add(OP_gradnorm_partial, A().b(ga).b(part).i(1).i(350).i(700));
    b.add(OP_gradnorm_partial, A().b(gb).b(part).i(2).i(0).i(300)); run(dv, b);
    run1(dv, OP_gradnorm_finish, A().b(part).i(3).b(nrm).f(1.0f));
    double s = 0; for (float x : g1) s += (double)x * x; for (float x : g2) s += (double)x * x;
    double rn = sqrt(s), coef = std::min(1.0, 1.0 / (rn + 1e-6)); auto hn = dv.down(nrm);
    check("gradnorm_partial+finish norm", {hn[1]}, {rn}, 1e-6);
    check("gradnorm clip coefficient", {hn[0]}, {coef}, 1e-6);
}

// ---- sampling ---------------------------------------------------------------------------------------------------------
static int sample_ref(const float* row, int V, double temp, int topk, int seed, int ctr, int rowi) {
    uint32_t c[4] = {(uint32_t)rowi, (uint32_t)ctr, 0, 0}, o[4]; philox_host(c, seed, 0x9E3779B9u, o);
    double u = (double)(float)((o[0] >> 8) * (1.0f / 16777216.0f));
    float mx = -INFINITY; for (int i = 0; i < V; i++) mx = std::max(mx, row[i]);
    float thr = -INFINITY;
    if (topk > 0 && topk < V) { std::vector<float> s(row, row + V); std::sort(s.begin(), s.end(), std::greater<float>()); thr = s[topk - 1]; }
    double Z = 0; for (int i = 0; i < V; i++) if (row[i] >= thr) Z += exp((double)(row[i] - mx) / temp);
    double run = 0, target = u * Z; int last = 0;
    for (int i = 0; i < V; i++) if (row[i] >= thr) { run += exp((double)(row[i] - mx) / temp); last = i; if (run > target) return i; }
    return last;
}
static void t_sample() {
    Rng r(101);
    {   // argmax: ties resolve to the lowest index; V=1; large V
        for (int V : {1, 257, 5000}) {
            int N = 6; Dev dv; int lg = dv.addf(N, V, V + 1), out = dv.addi(N); auto hl = rnd(r, N * V); 
            if (V > 10) { hl[0 * V + 5] = hl[0 * V + V - 1] = 100.f; hl[1 * V + V / 2] = hl[1 * V + 3] = 50.f; }
            dv.up(lg, hl);
            Batch b; b.add(OP_argmax, A().b(lg).b(out).i(0).i(3).i(V)); b.add(OP_sample, A().b(lg).b(out).b(-1).i(3).i(N).i(V).f(0.f).i(0).i(1));
            int e = run(dv, b); auto o = dv.downi(out); bool ok = e == 0;
            for (int i = 0; i < N; i++) { int bi = 0; for (int c = 1; c < V; c++) if (hl[i * V + c] > hl[i * V + bi]) bi = c; ok &= o[i] == bi; }
            check_true("argmax / sample(temp=0) V=" + std::to_string(V) + " lowest-index ties", ok);
        }
    }
    for (int topk : {0, 5, 40}) for (int V : {1, 100, 1000, 33000}) {
        int N = 32; Dev dv; int lg = dv.addf(N, V, V), out = dv.addi(N), ctr = dv.addi(1); auto hl = rnd(r, N * V, 2.0);
        dv.up(lg, hl); bool ok = true; double temp = 0.8;
        for (int ct : {0, 1, 7}) {
            dv.upi(ctr, {ct});
            Batch b; for (int r0 = 0; r0 < N; r0 += 8) b.add(OP_sample, A().b(lg).b(out).b(ctr).i(r0).i(r0 + 8).i(V).f((float)temp).i(topk).i(1234));
            ok &= run(dv, b) == 0; auto o = dv.downi(out);
            for (int i = 0; i < N; i++) ok &= o[i] == sample_ref(&hl[(size_t)i * V], V, (float)temp, topk, 1234, ct, i);
        }
        check_true("sample topk=" + std::to_string(topk) + " V=" + std::to_string(V) + " matches double CDF reference exactly (96 draws)", ok);
    }
    {   // distribution and determinism
        int N = 4096, V = 8; Dev dv; int lg = dv.addf(N, V), out = dv.addi(N), ctr = dv.addi(1); std::vector<float> hl(N * V);
        float base[8] = {0.f, 1.f, 2.f, -1.f, 0.5f, 1.5f, 3.f, -3.f}; for (int i = 0; i < N; i++) for (int c = 0; c < V; c++) hl[i * V + c] = base[c];
        dv.up(lg, hl); dv.upi(ctr, {3});
        Batch b; for (int r0 = 0; r0 < N; r0 += 512) b.add(OP_sample, A().b(lg).b(out).b(ctr).i(r0).i(r0 + 512).i(V).f(1.0f).i(0).i(77));
        run(dv, b); auto o = dv.downi(out); run(dv, b); check_true("sample bitwise reproducible", o == dv.downi(out));
        double Z = 0, p[8]; int cnt[8] = {0}; for (int c = 0; c < V; c++) { p[c] = exp((double)base[c]); Z += p[c]; } for (int x : o) cnt[x]++;
        bool ok = true; for (int c = 0; c < V; c++) { double pp = p[c] / Z, sd = sqrt(N * pp * (1 - pp)); ok &= fabs(cnt[c] - N * pp) < 5 * sd + 1; }
        check_true("sample empirical frequencies within 5 sigma of softmax (4096 draws)", ok);
    }
}

// ---- error codes ------------------------------------------------------------------------------------------------------
static void t_errors() {
    Dev dv; int f = dv.addf(4, 8), i = dv.addi(4), g = dv.addf(4, 8);
    int n = 4 * 8;
    check_true("gelu_fwd bad buf index -> err 1", run1(dv, OP_gelu_fwd, A().b(99).b(g).i(0).i(n)) == KVM_GPT_E_BUF);
    check_true("gelu_fwd int buffer passed as f32 -> err 2", run1(dv, OP_gelu_fwd, A().b(i).b(g).i(0).i(4)) == KVM_GPT_E_DTYPE);
    check_true("gelu_fwd range beyond buffer -> err 3", run1(dv, OP_gelu_fwd, A().b(f).b(g).i(0).i(n + 1)) == KVM_GPT_E_RANGE);
    check_true("attn_fwd unsupported hd -> err 4", run1(dv, OP_attn_fwd, A().b(f).b(f).b(f).b(g).b(g).i(0).i(0).i(0).i(4).i(1).i(1).i(48).f(1.f)) == KVM_GPT_E_SHAPE);
    auto before = dv.down(g);
    run1(dv, OP_gelu_fwd, A().b(99).b(g).i(0).i(n));
    check_true("failed op leaves outputs untouched", before == dv.down(g));
    Batch b; b.op.push_back(999); b.off.push_back(0); b.w.push_back(0);
    check_true("unknown op id -> flagged", run(dv, b) == 99);
}

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s, %d SMs, cc %d.%d\n", pr.name, pr.multiProcessorCount, pr.major, pr.minor);
    t_philox(); t_embedding(); t_matmul(); t_norms(); t_elementwise(); t_rope(); t_attention(); t_decode(); t_ce(); t_adamw(); t_sample(); t_errors();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
