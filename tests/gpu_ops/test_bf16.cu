// Per-op tests of real bf16 storage (KvmBuf dtype 2): cast_f32_bf16, cast_bf16_f32, matmul_tile_bf16s (include/kvm_ops_extra.cuh).
// Light build (own kernel dispatching only these ops + matmul_tile_bf16 as the reference): a few minutes of nvcc, not tens.
// Build: nvcc -O2 -arch=sm_80 -std=c++17 --fmad=false -Xcompiler -ffp-contract=off -I../../include -I. test_bf16.cu -o test_bf16
#define KVM_HARNESS_LIGHT 1
#include <algorithm>
#include <map>
#include <numeric>
#ifndef KVM_GPT_MAX_HD
#define KVM_GPT_MAX_HD 128
#endif
#include "harness.cuh"
#include "refs.hpp"
#include "kvm_ops_extra.cuh"

enum { OPX_cast_f32_bf16 = 122, OPX_cast_bf16_f32 = 123, OPX_matmul_tile_bf16s = 124, OPX_matmul_tile_bf16 = 114, OPX_round_bf16 = 113 };
__global__ void k_runx(KvmCtx c, const int32_t* w, const int* op, const int* off) {
    const int32_t* a = w + off[blockIdx.x];
    switch (op[blockIdx.x]) {
        case OPX_cast_f32_bf16: kvmextra::cast_f32_bf16(c, a); break;
        case OPX_cast_bf16_f32: kvmextra::cast_bf16_f32(c, a); break;
        case OPX_matmul_tile_bf16s: kvmextra::matmul_tile_bf16s(c, a); break;
        case OPX_matmul_tile_bf16: kvmextra::matmul_tile_bf16(c, a); break;
        case OPX_round_bf16: kvmextra::round_bf16(c, a); break;
        default: atomicCAS(c.err, 0, 99);
    }
}
struct BatchX {
    std::vector<int> op, off; std::vector<int32_t> w;
    void add(int id, const A& a) { op.push_back(id); off.push_back((int)w.size()); w.insert(w.end(), a.w.begin(), a.w.end()); }
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
struct MmCfg { int M, N, K; };
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

// ---- real bf16 storage (dtype 2) ------------------------------------------------------------------------------------------------------
static void t_bf16_storage() {
    Rng r(171);
    std::vector<float> v = {0.f, -0.f, INFINITY, -INFINITY, NAN, 1.f, -1.f, 3.4028235e38f, 3.38e38f, 1e-40f, 1.4e-45f,
                            f32bits(0x3f808000u), f32bits(0x3f818000u), f32bits(0x3f808001u), f32bits(0x7f7f8000u), f32bits(0x7fc12345u), f32bits(0x7f800001u)};
    for (int i = 0; i < 4000; i++) v.push_back(f32bits(r.u32() * 65537u + r.u32()));
    for (int i = 0; i < 1000; i++) v.push_back((float)(r.nrm() * 10));
    int n = (int)v.size(); Dev dv; int x = dv.addf(1, n), o = dv.addb(1, n), back = dv.addf(1, n); dv.up(x, v);
    int e = run1x(dv, OPX_cast_f32_bf16, A().b(x).b(o).i(0).i(n));
    auto raw = dv.downb_raw(o); bool ok = e == 0;
    for (int i = 0; i < n; i++) {
        float ref = bf16_ref(v[i]); uint32_t want = bitsf(ref) >> 16;
        bool same = std::isnan(ref) ? ((raw[i] & 0x7fff) > 0x7f80) : raw[i] == want;
        if (!same && ok) printf("  cast_f32_bf16 mismatch in %08x got %04x want %04x\n", bitsf(v[i]), raw[i], want);
        ok &= same;
    }
    check_true("cast_f32_bf16 bitwise vs independent RNE, 2-byte storage", ok);
    e = run1x(dv, OPX_cast_bf16_f32, A().b(o).b(back).i(0).i(n));
    auto g = dv.down(back); bool ok2 = e == 0;
    for (int i = 0; i < n; i++) { float ref = bf16_ref(v[i]); ok2 &= std::isnan(ref) ? std::isnan(g[i]) : bitsf(ref) == bitsf(g[i]); }
    check_true("cast_bf16_f32 widens exactly (round trip == round_bf16)", ok2);
    {   // partial range leaves the rest of the destination untouched
        Dev d2; int xs = d2.addf(1, 64), os = d2.addb(1, 64); std::vector<float> h(64); for (int i = 0; i < 64; i++) h[i] = 1.f + i;
        d2.up(xs, h); run1x(d2, OPX_cast_f32_bf16, A().b(xs).b(os).i(8).i(40));
        auto got = d2.downb_raw(os); bool okp = true; for (int i = 0; i < 64; i++) okp &= (i >= 8 && i < 40) ? got[i] == (bitsf(bf16_ref(h[i])) >> 16) : got[i] == 0;
        check_true("cast_f32_bf16 partial range", okp);
    }
    // matmul on bf16-storage operands == matmul_tile_bf16 on the widened operands, bit for bit (same tile code, same rounding)
    MmCfg shapes[] = {{1, 1, 1}, {5, 7, 3}, {64, 64, 32}, {65, 63, 33}, {130, 70, 100}, {1, 300, 17}, {70, 1, 90}, {64, 64, 512}};
    int combos = 0; bool allsame = true;
    // (A bf16, B bf16, ta, tb) combinations the op instantiates; every other one must be refused (tested below)
    auto supported = [](int sa, int sb, int ta, int tb) {
        return (sa && sb && !ta) || (!sa && sb && !(ta && tb)) || (sa && !sb && !tb);
    };
    for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 2; tb++) for (auto s : shapes) for (int mode = 1; mode < 4; mode++) for (int full = 0; full < 2; full++) {
        int M = s.M, N = s.N, K = s.K; bool sa = mode & 1, sb = mode & 2;
        if (!supported(sa, sb, ta, tb)) continue;
        Dev d;
        int ar = ta ? K : M, ac = ta ? M : K, br = tb ? N : K, bc = tb ? K : N;
        int ia = sa ? d.addb(ar, ac, ac + ((M + N) & 1 ? 1 : 3)) : d.addf(ar, ac, ac + 1);
        int ib = sb ? d.addb(br, bc, bc + 5) : d.addf(br, bc, bc + 2);
        int iaw = d.addf(ar, ac), ibw = d.addf(br, bc);     // widened reference operands
        int io = d.addf(M, N, N + 1), ioref = d.addf(M, N, N + 1);
        int ibias = d.addf(1, N), ires = d.addf(M, N), ipre = d.addf(M, N), iprer = d.addf(M, N);
        auto ha = rnd(r, ar * ac), hb = rnd(r, br * bc), ho = rnd(r, M * N), hbias = rnd(r, N), hres = rnd(r, M * N);
        if (sa) d.upb(ia, ha); else d.up(ia, ha);
        if (sb) d.upb(ib, hb); else d.up(ib, hb);
        d.up(iaw, sa ? d.downb(ia) : ha); d.up(ibw, sb ? d.downb(ib) : hb);
        d.up(io, ho); d.up(ioref, ho); d.up(ibias, hbias); d.up(ires, hres);
        int fl = ta | (tb << 1) | (full ? (4 | 8) : 0);
        BatchX b, br_;
        for (int m0 = 0; m0 < M; m0 += 64) for (int n0 = 0; n0 < N; n0 += 64) {
            b.add(OPX_matmul_tile_bf16s, A().b(ia).b(ib).b(io).b(full ? ibias : -1).b(full ? ires : -1).b(full ? ipre : -1).i(m0).i(n0).i(M).i(N).i(K).i(fl | (sa ? 128 : 0) | (sb ? 256 : 0)).f(0.9f));
            br_.add(OPX_matmul_tile_bf16, A().b(iaw).b(ibw).b(ioref).b(full ? ibias : -1).b(full ? ires : -1).b(full ? iprer : -1).i(m0).i(n0).i(M).i(N).i(K).i(fl).f(0.9f));
        }
        int e1 = runx(d, b), e2 = runx(d, br_);
        bool same = e1 == 0 && e2 == 0 && d.down(io) == d.down(ioref) && (!full || d.down(ipre) == d.down(iprer));
        if (!same && allsame) printf("  bf16s mismatch ta=%d tb=%d %dx%dx%d mode=%d full=%d e=%d/%d\n", ta, tb, M, N, K, mode, full, e1, e2);
        allsame &= same; combos++;
    }
    check_true("matmul_tile_bf16s bitwise == matmul_tile_bf16 on widened operands (" + S(combos) + " combos: ta/tb, A/B/both bf16, epilogue, padded ld)", allsame);
    {   // error paths: wrong dtypes / flags are reported, memory untouched
        Dev d; int fa = d.addf(8, 8), fb = d.addf(8, 8), ba = d.addb(8, 8), bb = d.addb(8, 8), fo = d.addf(8, 8), bo = d.addb(8, 8);
        std::vector<float> h(64, 2.f); d.up(fo, h);
        auto mm = [&](int a, int b, int out, int fl) { return run1x(d, OPX_matmul_tile_bf16s, A().b(a).b(b).b(out).b(-1).b(-1).b(-1).i(0).i(0).i(8).i(8).i(8).i(fl).f(1.f)); };
        check_true("bf16s flag 128 on an f32 buffer is refused", mm(fa, bb, fo, 128 | 256) != 0);
        check_true("bf16s uninstantiated combination (A and B bf16, A transposed) is refused", mm(ba, bb, fo, 128 | 256 | 1) != 0);
        check_true("bf16s bf16 buffer without its flag is refused", mm(ba, bb, fo, 128) != 0);
        check_true("bf16s with no bf16 flag is refused", mm(fa, fb, fo, 0) != 0);
        check_true("bf16s flag 16 is refused", mm(ba, bb, fo, 128 | 256 | 16) != 0);
        check_true("bf16s output must be f32", mm(ba, bb, bo, 128 | 256) != 0);
        check_true("bf16s refused calls left the output untouched", d.down(fo) == h);
        check_true("f32-only op refuses a bf16 buffer", run1x(d, OPX_round_bf16, A().b(ba).b(fo).i(0).i(8)) != 0);
        check_true("cast_f32_bf16 refuses an f32 destination", run1x(d, OPX_cast_f32_bf16, A().b(fa).b(fb).i(0).i(8)) != 0);
        check_true("cast_f32_bf16 beyond the buffer", run1x(d, OPX_cast_f32_bf16, A().b(fa).b(ba).i(0).i(65)) != 0);
    }
}


int main() {
    t_bf16_storage();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
