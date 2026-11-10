// bf16 tensor-core GEMM (matmul_mma_bf16) and the bf16-STORAGE paths of the GPT-pack ops against independent float64 CPU references.
//   matmul_mma_bf16: all four transpose combinations, ragged shapes (fast path and the matmul_tile_bf16 fallback), every out/res/pre storage mix.
//   storage-aware ops: embedding_fwd, layernorm/rmsnorm fwd + bwd, norm_bwd_params, colsum, gelu_bwd, residual_add, rope fwd/bwd, softmax_ce fwd/bwd:
//   inputs rounded to bf16, outputs fp32 (tolerance of the fp32 op) or bf16 (one bf16 ulp relative to the output scale), plus error paths.
// Build: nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -I../../include -I. test_bf16.cu -o test_bf16
#include <algorithm>
#ifndef KVM_GPT_MAX_HD
#define KVM_GPT_MAX_HD 64
#endif
#define KVM_HARNESS_NO_FULL   // this test dispatches only the ops it exercises (the all-ops kernel of the harness costs minutes of compile time)
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
__global__ void __launch_bounds__(256, 1) k_runx(KvmCtx c, const int32_t* w, const int* op, const int* off) {
    const int32_t* a = w + off[blockIdx.x];
    switch (op[blockIdx.x]) {
        case OPX_matmul_mma_bf16: kvmextra::matmul_mma_bf16(c, a); break;
        case OP_embedding_fwd: kvmgpt::embedding_fwd(c, a); break;
        case OP_layernorm_fwd: kvmgpt::layernorm_fwd(c, a); break;
        case OP_rmsnorm_fwd: kvmgpt::rmsnorm_fwd(c, a); break;
        case OP_layernorm_bwd_dx: kvmgpt::layernorm_bwd_dx(c, a); break;
        case OP_rmsnorm_bwd_dx: kvmgpt::rmsnorm_bwd_dx(c, a); break;
        case OP_norm_bwd_params: kvmgpt::norm_bwd_params(c, a); break;
        case OP_colsum: kvmgpt::colsum(c, a); break;
        case OP_gelu_bwd: kvmgpt::gelu_bwd(c, a); break;
        case OP_residual_add: kvmgpt::residual_add(c, a); break;
        case OP_rope_fwd: kvmgpt::rope_fwd(c, a); break;
        case OP_rope_bwd: kvmgpt::rope_bwd(c, a); break;
        case OP_softmax_ce_fwd: kvmgpt::softmax_ce_fwd(c, a); break;
        case OP_softmax_ce_bwd: kvmgpt::softmax_ce_bwd(c, a); break;
        default: atomicCAS(c.err, 0, 99);
    }
}
struct BatchX {
    std::vector<int> op, off; std::vector<int32_t> w;
    void add(int id, const A& a) { op.push_back(id); off.push_back((int)w.size()); w.insert(w.end(), a.w.begin(), a.w.end()); }
};
static int g_attr = 0;
static int runx(Dev& d, const BatchX& b) {
    const int smem = std::max(kvm_gpt_smem_bytes(KVM_GPT_MAX_HD), KVM_MMA_SMEM);
    if (!g_attr) { CK(cudaFuncSetAttribute(k_runx, cudaFuncAttributeMaxDynamicSharedMemorySize, std::max(112 * 1024, (int)kvm_gpt_smem_bytes(KVM_GPT_MAX_HD)))); kvm_hopper_host_init(); g_attr = 1; }
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
static int run1x(Dev& d, int id, const A& a) { BatchX b; b.add(id, a); return runx(d, b); }

static std::string S(int v) { return std::to_string(v); }
static float bf16_ref(float x) {   // independent RNE to the bf16 grid (no bit tricks)
    if (!std::isfinite(x) || x == 0.f) return x;
    int e; std::frexp(x, &e);
    const double q = std::ldexp(1.0, e - 8);           // 8 significant bits
    return (float)(std::rint((double)x / q) * q);
}
static D rb(const std::vector<float>& v) { D o(v.size()); for (size_t i = 0; i < v.size(); i++) o[i] = bf16_ref(v[i]); return o; }
static std::vector<float> rbf(std::vector<float> v) { for (auto& x : v) x = bf16_ref(x); return v; }
// stores host floats (made bf16-representable if the buffer is bf16) and returns what the device holds: the references use these values
static int mk(Dev& dv, bool bf, int rows, int cols, int ld = 0) { return bf ? dv.addb(rows, cols, ld) : dv.addf(rows, cols, ld); }
static std::vector<float> put(Dev& dv, int i, bool bf, std::vector<float> h) { if (bf) { h = rbf(h); dv.upb(i, h); } else dv.up(i, h); return h; }
static std::vector<float> get(Dev& dv, int i, bool bf) { return bf ? dv.downb(i) : dv.down(i); }

// ---- matmul_mma_bf16 -------------------------------------------------------------------------------------------------------------------
static void t_mma() {
    Rng r(2024);
    struct Cfg { int M, N, K; };
    const bool small = getenv("TEST_BF16_SMALL") != nullptr;   // sanitizer runs: fewer, smaller shapes
    const std::vector<Cfg> shapes = small ? std::vector<Cfg>{{130, 260, 40}, {1, 1, 8}, {5, 7, 3}, {17, 33, 24}, {129, 9, 72}}
                                          : std::vector<Cfg>{{128, 256, 32}, {130, 260, 40}, {1, 1, 8}, {200, 300, 96}, {64, 64, 16}, {5, 7, 3}, {256, 512, 128}, {17, 33, 24}, {131, 9, 1000}, {300, 513, 72}};
    int combos = 0; double worst = 0;
    for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) for (auto s : shapes)
        for (int mode = 0; mode < 6; mode++) {   // 0 plain bf16 out; 1 plain f32 out; 2 full bf16; 3 full f32 out, bf16 res/pre; 4 full, f32 res/pre; 5 fp32 operands (fallback)
            const int M = s.M, N = s.N, K = s.K;
            const bool full = mode >= 2, fo = mode == 1 || mode == 4, fr = mode == 4, fp = mode == 4 && false, fops = mode == 5;
            const bool obf = mode == 0 || mode == 2 || mode == 3, rbf_ = mode != 4, pbf = mode != 4;
            (void)fo; (void)fr; (void)fp;
            Dev dv;
            const int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
            const int pad = (M + N + K) % 3 == 0 ? 0 : 8;   // sometimes ld == cols; ragged cases below make the fast path ineligible
            const bool abf = !fops;
            int ia = mk(dv, abf, ar, ac, ac + pad), ib = mk(dv, abf, br, bc, bc + pad), io = mk(dv, obf, M, N, N + 8);
            int ibias = dv.addf(1, N), ires = mk(dv, rbf_, M, N, N), ipre = mk(dv, pbf, M, N, N + 8);
            auto ha = put(dv, ia, abf, rnd(r, ar * ac)), hb = put(dv, ib, abf, rnd(r, br * bc));
            auto hbias = rnd(r, N), hres = rnd(r, M * N), ho = rnd(r, M * N);
            dv.up(ibias, hbias); put(dv, ires, rbf_, hres); put(dv, io, obf, ho);
            hres = rbf_ ? rbf(hres) : hres; ho = obf ? rbf(ho) : ho;
            const float alpha = full ? 0.7f : 1.f;
            const int fl = ta | (tb << 1) | (full ? (4 | 8) : 0);
            BatchX b;
            for (int m0 = 0; m0 < M; m0 += KVM_MMA_BM) for (int n0 = 0; n0 < N; n0 += KVM_MMA_BN)
                b.add(OPX_matmul_mma_bf16, A().b(ia).b(ib).b(io).b(full ? ibias : -1).b(full ? ires : -1).b(full ? ipre : -1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(alpha));
            const int e = runx(dv, b);
            D C = mm_d(rb(ha), rb(hb), M, N, K, ta, tb), ref(M * N), pre(M * N);
            for (int i = 0; i < M * N; i++) {
                double v = alpha * C[i];
                if (full) { v += hbias[i % N]; pre[i] = v; v = gelu_d(v) + hres[i]; v += ho[i]; }
                ref[i] = v;
            }
            const std::string nm = std::string("mma_bf16 ta=") + S(ta) + " tb=" + S(tb) + " " + S(M) + "x" + S(N) + "x" + S(K) + " mode=" + S(mode);
            auto go = get(dv, io, obf);
            worst = std::max(worst, relerr(go, ref)); combos++;
            check(nm, go, ref, obf ? 1.2e-2 : 1e-4);
            if (full) check(nm + " pre", get(dv, ipre, pbf), pre, pbf ? 1.2e-2 : 1e-4);
            if (e) check_true(nm + " err", false, ("device error " + S(e)).c_str());
        }
    printf("  matmul_mma_bf16: %d configurations, worst relerr %.3e\n", combos, worst);
    {   // bf16-stored results are exactly representable (the low 16 bits of every stored word are zero after widening is trivially true; check the grid)
        Dev dv; int M = 128, N = 256, K = 64;
        int ia = dv.addb(M, K), ib = dv.addb(K, N), io = dv.addb(M, N);
        put(dv, ia, true, rnd(r, M * K)); put(dv, ib, true, rnd(r, K * N));
        int e = run1x(dv, OPX_matmul_mma_bf16, A().b(ia).b(ib).b(io).b(-1).b(-1).b(-1).i(0).i(0).i(M).i(N).i(K).i(0).f(1.f));
        auto g = dv.downb(io); bool grid = true; for (float x : g) grid &= bf16_ref(x) == x;
        check_true("mma_bf16 stores only bf16-representable values", grid && e == 0);
    }
    {   // errors: flags beyond 15, m0 >= M, K beyond A, fp32 operands accepted (fallback), i32 operand rejected
        Dev dv; int ia = dv.addb(8, 8), ib = dv.addb(8, 8), io = dv.addb(8, 8), ii = dv.addi(64);
        check_true("mma_bf16 flags 16 -> E_SHAPE", run1x(dv, OPX_matmul_mma_bf16, A().b(ia).b(ib).b(io).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(8).i(16).f(1.f)) == KVM_GPT_E_SHAPE);
        check_true("mma_bf16 m0 >= M -> E_RANGE", run1x(dv, OPX_matmul_mma_bf16, A().b(ia).b(ib).b(io).b(-1).b(-1).b(-1).i(8).i(0).i(8).i(8).i(8).i(0).f(1.f)) == KVM_GPT_E_RANGE);
        check_true("mma_bf16 K beyond A -> E_RANGE", run1x(dv, OPX_matmul_mma_bf16, A().b(ia).b(ib).b(io).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(9).i(0).f(1.f)) == KVM_GPT_E_RANGE);
        check_true("mma_bf16 i32 operand -> E_DTYPE", run1x(dv, OPX_matmul_mma_bf16, A().b(ii).b(ib).b(io).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(8).i(0).f(1.f)) == KVM_GPT_E_DTYPE);
    }
}

// ---- storage-aware GPT-pack ops --------------------------------------------------------------------------------------------------------
// tolerance: bf16 output = one bf16 ulp (2^-8) of the output scale plus the fp32 op's own error; fp32 output = the fp32 op's
static double tol_of(bool bf) { return bf ? 6e-3 : 2e-5; }
static void t_ops() {
    Rng r(77);
    for (int mask = 0; mask < 4; mask++) {   // bit 0: input storage bf16, bit 1: output storage bf16
        const bool ib = mask & 1, ob = mask & 2;
        const std::string tag = std::string(" [in ") + (ib ? "bf16" : "f32") + ", out " + (ob ? "bf16" : "f32") + "]";
        for (int d : getenv("TEST_BF16_SMALL") ? std::vector<int>{20, 22, 96} : std::vector<int>{20, 22, 96, 768}) {   // norms: 4-aligned d (vector path), d % 4 != 0 (scalar path)
            const int R = 9;
            for (int rms = 0; rms < 2; rms++) {
                Dev dv;
                int x = mk(dv, ib, R, d, d + 4), g = dv.addf(1, d), be = dv.addf(1, d), o = mk(dv, ob, R, d, d + 4), mean = dv.addf(1, R), rstd = dv.addf(1, R);
                auto hx = put(dv, x, ib, rnd(r, R * d, 2.0)), hg = rnd(r, d), hb = rnd(r, d);
                dv.up(g, hg); dv.up(be, hb);
                BatchX bb;
                for (int r0 = 0; r0 < R; r0 += 4) {
                    if (rms) bb.add(OP_rmsnorm_fwd, A().b(x).b(g).b(o).b(rstd).i(r0).i(std::min(R, r0 + 4)).i(d).f(1e-5f));
                    else bb.add(OP_layernorm_fwd, A().b(x).b(g).b(be).b(o).b(mean).b(rstd).i(r0).i(std::min(R, r0 + 4)).i(d).f(1e-5f));
                }
                int e = runx(dv, bb);
                D m, rs; D ref = norm_fwd_d(dbl(hx), dbl(hg), rms ? D() : dbl(hb), R, d, 1e-5, rms, m, rs);
                check(std::string(rms ? "rmsnorm_fwd" : "layernorm_fwd") + " d=" + S(d) + tag, get(dv, o, ob), ref, tol_of(ob));
                check_true(std::string(rms ? "rmsnorm_fwd" : "layernorm_fwd") + " d=" + S(d) + tag + " err==0", e == 0);
                // backward: dx and parameter gradients
                int dy = mk(dv, ib, R, d, d + 4), dx = mk(dv, ob, R, d, d + 4), dg = dv.addf(1, d), db = dv.addf(1, d);
                auto hdy = put(dv, dy, ib, rnd(r, R * d));
                BatchX b2, b3;
                for (int r0 = 0; r0 < R; r0 += 4) {
                    if (rms) b2.add(OP_rmsnorm_bwd_dx, A().b(x).b(dy).b(g).b(rstd).b(dx).i(r0).i(std::min(R, r0 + 4)).i(d).i(0));
                    else b2.add(OP_layernorm_bwd_dx, A().b(x).b(dy).b(g).b(mean).b(rstd).b(dx).i(r0).i(std::min(R, r0 + 4)).i(d).i(0));
                }
                for (int c0 = 0; c0 < d; c0 += 100) b3.add(OP_norm_bwd_params, A().b(x).b(dy).b(rms ? -1 : mean).b(rstd).b(dg).b(rms ? -1 : db).i(R).i(d).i(c0).i(std::min(d, c0 + 100)).i(0));
                int e2 = runx(dv, b2) | runx(dv, b3);
                D rdx, rdg, rdb; norm_bwd_d(dbl(hx), dbl(hdy), dbl(hg), R, d, 1e-5, rms, rdx, rdg, rdb);
                check(std::string(rms ? "rmsnorm_bwd_dx" : "layernorm_bwd_dx") + " d=" + S(d) + tag, get(dv, dx, ob), rdx, tol_of(ob));
                check(std::string("norm_bwd_params dgamma d=") + S(d) + tag, dv.down(dg), rdg, 3e-5);
                if (!rms) check(std::string("norm_bwd_params dbeta d=") + S(d) + tag, dv.down(db), rdb, 3e-5);
                check_true(std::string("norm bwd d=") + S(d) + tag + " err==0", e2 == 0);
            }
        }
        {   // colsum, gelu_bwd, residual_add, embedding_fwd, rope
            const int R = 37, C = 70, n = R * C;
            Dev dv;
            int cx = mk(dv, ib, R, C, C + 2), co = dv.addf(1, C);
            auto hcx = put(dv, cx, ib, rnd(r, n));
            BatchX b; for (int c0 = 0; c0 < C; c0 += 32) b.add(OP_colsum, A().b(cx).b(co).i(R).i(c0).i(std::min(C, c0 + 32)).i(0));
            int e0 = runx(dv, b); D ref(C, 0); D hd = dbl(hcx);
            for (int i = 0; i < n; i++) ref[i % C] += hd[i];
            check("colsum" + tag, dv.down(co), ref, 3e-5); check_true("colsum err==0" + tag, e0 == 0);

            int px = mk(dv, ib, 1, n), pdy = mk(dv, ib, 1, n), pdx = mk(dv, ob, 1, n);
            auto hpx = put(dv, px, ib, rnd(r, n, 2.0)), hpdy = put(dv, pdy, ib, rnd(r, n)); auto hdx0 = put(dv, pdx, ob, rnd(r, n));
            for (int acc = 0; acc < 2; acc++) {
                put(dv, pdx, ob, hdx0);
                BatchX bg; for (int e = 0; e < n; e += 300) bg.add(OP_gelu_bwd, A().b(px).b(pdy).b(pdx).i(e).i(std::min(n, e + 300)).i(acc));
                int eg = runx(dv, bg); D rg(n);
                const D dyv = dbl(hpdy), xv = dbl(hpx), d0v = dbl(hdx0);
                for (int i = 0; i < n; i++) rg[i] = dyv[i] * gelu_grad_d(xv[i]) + (acc ? d0v[i] : 0.0);
                check(std::string("gelu_bwd acc=") + S(acc) + tag, get(dv, pdx, ob), rg, tol_of(ob)); check_true("gelu_bwd err==0" + tag, eg == 0);
            }
            int ra = mk(dv, ib, 1, n), rb2 = mk(dv, ib, 1, n), ro = mk(dv, ob, 1, n);
            auto hra = put(dv, ra, ib, rnd(r, n)), hrb = put(dv, rb2, ib, rnd(r, n));
            BatchX br; for (int e = 0; e < n; e += 500) br.add(OP_residual_add, A().b(ra).b(rb2).b(ro).i(e).i(std::min(n, e + 500)));
            int er = runx(dv, br); D rr(n); const D ra_ = dbl(hra), rb_ = dbl(hrb); for (int i = 0; i < n; i++) rr[i] = ra_[i] + rb_[i];
            check("residual_add" + tag, get(dv, ro, ob), rr, tol_of(ob)); check_true("residual_add err==0" + tag, er == 0);

            const int V = 50, dd = 20, N = 11, T = 8;
            int ids = dv.addi(N), wte = dv.addf(V, dd), wpe = dv.addf(T, dd), eo = mk(dv, ob, N, dd, dd + 4);
            std::vector<int32_t> hid(N); for (auto& v : hid) v = r.u32() % V;
            auto hw = rnd(r, V * dd), hp = rnd(r, T * dd); dv.upi(ids, hid); dv.up(wte, hw); dv.up(wpe, hp);
            int ee = run1x(dv, OP_embedding_fwd, A().b(ids).b(wte).b(wpe).b(eo).i(0).i(N).i(T).i(dd));
            D re(N * dd); for (int i = 0; i < N; i++) for (int c = 0; c < dd; c++) re[i * dd + c] = (double)hw[hid[i] * dd + c] + hp[(i % T) * dd + c];
            check("embedding_fwd" + tag, get(dv, eo, ob), re, tol_of(ob)); check_true("embedding_fwd err==0" + tag, ee == 0);

            const int H = 3, hd2 = 16, RT = 13, TT = 8;
            int qx = mk(dv, ib, RT, H * hd2, H * hd2 + 4), qo = mk(dv, ob, RT, H * hd2, H * hd2 + 4);
            auto hqx = put(dv, qx, ib, rnd(r, RT * H * hd2));
            for (int sign = 0; sign < 2; sign++) {
                BatchX bq; for (int r0 = 0; r0 < RT; r0 += 5) bq.add(sign ? OP_rope_bwd : OP_rope_fwd, A().b(qx).b(qo).b(-1).i(r0).i(std::min(RT, r0 + 5)).i(TT).i(H).i(hd2).f(10000.f));
                int eq = runx(dv, bq); D rq(RT * H * hd2); D hxd = dbl(hqx);
                for (int rr_ = 0; rr_ < RT; rr_++) for (int h = 0; h < H; h++) for (int i = 0; i < hd2 / 2; i++) {
                    double ang = (double)(rr_ % TT) * pow(10000.0, -(double)(2 * i) / hd2), c = cos(ang), s = sin(ang) * (sign ? -1 : 1);
                    double x1 = hxd[rr_ * H * hd2 + h * hd2 + i], x2 = hxd[rr_ * H * hd2 + h * hd2 + i + hd2 / 2];
                    rq[rr_ * H * hd2 + h * hd2 + i] = x1 * c - x2 * s; rq[rr_ * H * hd2 + h * hd2 + i + hd2 / 2] = x1 * s + x2 * c;
                }
                check(std::string(sign ? "rope_bwd" : "rope_fwd") + tag, get(dv, qo, ob), rq, tol_of(ob) + (ob ? 0 : 3e-5)); check_true("rope err==0" + tag, eq == 0);
            }
        }
        for (int V : getenv("TEST_BF16_SMALL") ? std::vector<int>{50, 64, 130} : std::vector<int>{50, 64, 130, 4101}) {   // cross entropy: ragged V (scalar path) and 4-aligned V (vector path)
            const int N = 6; const int ign = -100;
            Dev dv;
            int lg = mk(dv, ib, N, V, V + (V % 4 ? 3 : 0)), tg = dv.addi(N), rl = dv.addf(1, N), ls = dv.addf(1, N), loss = dv.addf(1, 2), dl = mk(dv, ob, N, V, V + (V % 4 ? 3 : 0));
            auto hl = put(dv, lg, ib, rnd(r, N * V, 3.0));
            std::vector<int32_t> ht(N); for (int i = 0; i < N; i++) ht[i] = r.u32() % V; ht[2] = ign;
            dv.upi(tg, ht);
            BatchX b; for (int r0 = 0; r0 < N; r0 += 2) b.add(OP_softmax_ce_fwd, A().b(lg).b(tg).b(rl).b(ls).i(r0).i(std::min(N, r0 + 2)).i(V).i(ign));
            int e1 = runx(dv, b);
            D dref; std::vector<int> ti(ht.begin(), ht.end()); double lref = ce_d(dbl(hl), ti, N, V, ign, &dref);
            auto lsv = dv.down(ls), rlv = dv.down(rl); double s = 0; int cnt = 0; for (int i = 0; i < N; i++) if (ht[i] != ign) { s += rlv[i]; cnt++; }
            check_true(std::string("softmax_ce_fwd V=") + S(V) + tag + " loss", fabs(s / cnt - lref) <= 2e-5 * std::max(1.0, fabs(lref)) && e1 == 0, ("got " + std::to_string(s / cnt) + " ref " + std::to_string(lref)).c_str());
            dv.up(loss, {(float)lref, 1.f / (float)cnt});
            BatchX b2; for (int r0 = 0; r0 < N; r0 += 2) b2.add(OP_softmax_ce_bwd, A().b(lg).b(tg).b(ls).b(loss).b(dl).i(r0).i(std::min(N, r0 + 2)).i(V).i(ign).f(1.f));
            int e2 = runx(dv, b2);
            check(std::string("softmax_ce_bwd V=") + S(V) + tag, get(dv, dl, ob), dref, ob ? 8e-3 : 3e-5); check_true("softmax_ce_bwd err==0" + tag, e2 == 0);
        }
    }
}

int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("device: %s, %d SMs, cc %d.%d, mma smem %d B\n", pr.name, pr.multiProcessorCount, pr.major, pr.minor, KVM_MMA_SMEM);
    t_mma(); t_ops();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
