#pragma once
// Shared harness for the standalone op tests: device buffer table, typed-argument builder, a one-block-per-task launcher
// (no scheduler), error accounting and CPU-reference comparison. No dependency on the executor.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "kvm_ctx_shim.cuh"
#include "kvm_ops_gpt.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct OpInfo { const char* name; int id; const char* sig; };
static const OpInfo OPS[] = {
#define X(n, id, s) {#n, id, s},
#include "kvm_ops_gpt.def"
#undef X
};
enum {
#define X(n, id, s) OP_##n = id,
#include "kvm_ops_gpt.def"
#undef X
};

struct A {   // typed argument words
    std::vector<int32_t> w;
    A& b(int idx) { w.push_back(idx); return *this; }
    A& i(int v) { w.push_back(v); return *this; }
    A& f(float v) { int32_t x; memcpy(&x, &v, 4); w.push_back(x); return *this; }
};

struct Dev {
    std::vector<KvmBuf> tab;
    int32_t* err = nullptr;
    KvmBuf* tab_d = nullptr;
    Dev() { CK(cudaMalloc(&err, 4)); CK(cudaMemset(err, 0, 4)); }
    ~Dev() { for (auto& b : tab) cudaFree(b.p); cudaFree(err); cudaFree(tab_d); }
    int add(int rows, int cols, int ld, int dtype) {
        if (!ld) ld = cols;
        void* p; CK(cudaMalloc(&p, (size_t)rows * ld * 4 + 16)); CK(cudaMemset(p, 0, (size_t)rows * ld * 4 + 16));
        KvmBuf b{}; b.p = p; b.rows = rows; b.cols = cols; b.ld = ld; b.dtype = dtype; tab.push_back(b);
        return (int)tab.size() - 1;
    }
    int addf(int rows, int cols, int ld = 0) { return add(rows, cols, ld, 0); }
    int addi(int n) { return add(1, n, 0, 1); }
    int addb(int rows, int cols, int ld = 0) { return add(rows, cols, ld, 2); }   // raw bf16 storage (KvmBuf.dtype 2)
    void upb(int i, const std::vector<float>& h) {   // host floats must already be bf16-representable: stores the top 16 bits
        auto& b = tab[i]; std::vector<uint16_t> t((size_t)b.rows * b.ld, 0);
        for (int r = 0; r < b.rows; r++) for (int c = 0; c < b.cols; c++) { uint32_t u; memcpy(&u, &h[(size_t)r * b.cols + c], 4); t[(size_t)r * b.ld + c] = (uint16_t)(u >> 16); }
        CK(cudaMemcpy(b.p, t.data(), t.size() * 2, cudaMemcpyHostToDevice));
    }
    std::vector<float> downb(int i, std::vector<uint16_t>* raw = nullptr) {
        auto& b = tab[i]; std::vector<uint16_t> t((size_t)b.rows * b.ld); std::vector<float> o((size_t)b.rows * b.cols);
        CK(cudaMemcpy(t.data(), b.p, t.size() * 2, cudaMemcpyDeviceToHost));
        for (int r = 0; r < b.rows; r++) for (int c = 0; c < b.cols; c++) { uint32_t u = (uint32_t)t[(size_t)r * b.ld + c] << 16; memcpy(&o[(size_t)r * b.cols + c], &u, 4); }
        if (raw) *raw = t;
        return o;
    }
    void up(int i, const std::vector<float>& h) {
        auto& b = tab[i]; std::vector<float> t((size_t)b.rows * b.ld, 0.f);
        for (int r = 0; r < b.rows; r++) for (int c = 0; c < b.cols; c++) t[(size_t)r * b.ld + c] = h[(size_t)r * b.cols + c];
        CK(cudaMemcpy(b.p, t.data(), t.size() * 4, cudaMemcpyHostToDevice));
    }
    void upi(int i, const std::vector<int32_t>& h) { CK(cudaMemcpy(tab[i].p, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
    std::vector<float> down(int i) {
        auto& b = tab[i]; std::vector<float> t((size_t)b.rows * b.ld), o((size_t)b.rows * b.cols);
        CK(cudaMemcpy(t.data(), b.p, t.size() * 4, cudaMemcpyDeviceToHost));
        for (int r = 0; r < b.rows; r++) for (int c = 0; c < b.cols; c++) o[(size_t)r * b.cols + c] = t[(size_t)r * b.ld + c];
        return o;
    }
    std::vector<int32_t> downi(int i) { std::vector<int32_t> o(tab[i].cols); CK(cudaMemcpy(o.data(), tab[i].p, o.size() * 4, cudaMemcpyDeviceToHost)); return o; }
    size_t tab_cap = 0;
    // Uploads the buffer table. The device table keeps its address while the buffer count is unchanged, so contexts captured
    // earlier (staged launches) stay valid; add all buffers before staging.
    void sync_table() {
        if (tab_d && tab_cap != tab.size()) { cudaFree(tab_d); tab_d = nullptr; }
        if (!tab_d) { CK(cudaMalloc(&tab_d, tab.size() * sizeof(KvmBuf))); tab_cap = tab.size(); }
        CK(cudaMemcpy(tab_d, tab.data(), tab.size() * sizeof(KvmBuf), cudaMemcpyHostToDevice));
    }
    int geterr() { int e; CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost)); CK(cudaMemset(err, 0, 4)); return e; }
};

#ifndef BENCH_MINB
#define BENCH_MINB 1   // -DBENCH_MINB=2 caps the all-ops kernel at 128 registers (two blocks per SM), as an executor could
#endif
__global__ void __launch_bounds__(KVM_GPT_THREADS, BENCH_MINB) k_run(KvmCtx c, const int32_t* w, const int* op, const int* off) {
    if (!kvm_gpt_dispatch(c, op[blockIdx.x], w + off[blockIdx.x])) atomicCAS(c.err, 0, 99);
}

struct Batch {   // independent tasks, one block each
    std::vector<int> op, off; std::vector<int32_t> w;
    void add(int id, const A& a) {
        const OpInfo* oi = nullptr;
        for (auto& o : OPS) if (o.id == id) oi = &o;
        int ar = 0; for (const char* p = oi->sig; *p; p++) ar += *p == ':';
        if (ar != (int)a.w.size()) { printf("arity mismatch for %s: def %d, got %d\n", oi->name, ar, (int)a.w.size()); exit(2); }
        op.push_back(id); off.push_back((int)w.size()); w.insert(w.end(), a.w.begin(), a.w.end());
    }
};
static int g_smem_attr = 0;
// Runs the batch (one block of 256 threads per task) and returns the device error word (0 = ok).
static int run(Dev& d, const Batch& b, int smem = kvm_gpt_smem_bytes(KVM_GPT_MAX_HD)) {   // what the executor would launch with
    if (!g_smem_attr) { CK(cudaFuncSetAttribute(k_run, cudaFuncAttributeMaxDynamicSharedMemorySize, 112 * 1024)); g_smem_attr = 1; }
    d.sync_table();
    int32_t* w; int *op, *off;
    CK(cudaMalloc(&w, b.w.size() * 4 + 4)); CK(cudaMalloc(&op, b.op.size() * 4)); CK(cudaMalloc(&off, b.off.size() * 4));
    CK(cudaMemcpy(w, b.w.data(), b.w.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(op, b.op.data(), b.op.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(off, b.off.data(), b.off.size() * 4, cudaMemcpyHostToDevice));
    KvmCtx c{d.tab_d, (int32_t)d.tab.size(), d.err};
    k_run<<<(unsigned)b.op.size(), KVM_GPT_THREADS, smem>>>(c, w, op, off);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    cudaFree(w); cudaFree(op); cudaFree(off);
    return d.geterr();
}
static int run1(Dev& d, int id, const A& a, int smem = kvm_gpt_smem_bytes(KVM_GPT_MAX_HD)) { Batch b; b.add(id, a); return run(d, b, smem); }

// ---- accounting / comparison ------------------------------------------------------------------------------------------
static int g_pass = 0, g_fail = 0;
static void verdict(const std::string& name, bool ok, const char* detail) {
    printf("%s  %s  %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail);
    (ok ? g_pass : g_fail)++;
}
// max |got-ref| / (max|ref| + tiny) <= tol (relative to the reference's scale); also fails on NaN/Inf.
static double relerr(const std::vector<float>& got, const std::vector<double>& ref, double floor_ = 1e-30) {
    double mx = floor_, me = 0; bool bad = got.size() != ref.size();
    for (double r : ref) mx = std::max(mx, fabs(r));
    for (size_t i = 0; i < ref.size() && !bad; i++) { if (!std::isfinite(got[i])) bad = true; me = std::max(me, fabs(got[i] - ref[i])); }
    return bad ? 1e30 : me / mx;
}
// floor_: lower bound on the reference scale, for outputs that are exactly zero mathematically (d=1 norms, T=1 attention).
static void check(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref, double tol, double floor_ = 1e-30) {
    double e = relerr(got, ref, floor_); char d[96]; snprintf(d, sizeof d, "relerr %.3e (tol %.0e)", e, tol);
    verdict(name, e <= tol, d);
}
static void check_true(const std::string& name, bool ok, const char* detail = "") { verdict(name, ok, detail); }

// ---- randomness / conversions -----------------------------------------------------------------------------------------
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 2862933555777941757ull + 3037000493ull) {}
    uint32_t u32() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 16); }
    double uni() { return (u32() + 0.5) / 4294967296.0; }
    double nrm() { return sqrt(-2 * log(uni())) * cos(6.283185307179586 * uni()); }
};
static std::vector<float> rnd(Rng& r, size_t n, double sc = 1.0) { std::vector<float> v(n); for (auto& x : v) x = (float)(r.nrm() * sc); return v; }
static std::vector<double> dbl(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

// Philox-4x32-10 on the host (known-answer tested).
static void philox_host(uint32_t c[4], uint32_t k0, uint32_t k1, uint32_t o[4]) {
    uint32_t x0 = c[0], x1 = c[1], x2 = c[2], x3 = c[3];
    for (int i = 0; i < 10; i++) {
        uint64_t p0 = 0xD2511F53ull * x0, p1 = 0xCD9E8D57ull * x2;
        uint32_t n0 = (uint32_t)(p1 >> 32) ^ x1 ^ k0, n1 = (uint32_t)p1, n2 = (uint32_t)(p0 >> 32) ^ x3 ^ k1, n3 = (uint32_t)p0;
        x0 = n0; x1 = n1; x2 = n2; x3 = n3; k0 += 0x9E3779B9u; k1 += 0xBB67AE85u;
    }
    o[0] = x0; o[1] = x1; o[2] = x2; o[3] = x3;
}
