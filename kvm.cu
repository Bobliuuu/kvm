// kvm: "kittens"-style GPU virtual machine, v2 prototype (sm_80).
//
// ONE persistent kernel interprets an instruction tape. The tape encodes an ENTIRE training run
// (every step: fwd, loss, bwd, SGD) of a 2-layer MLP. No kernel boundaries between ops or steps:
// instructions synchronise through monotonic global counters (fine-grained, per row-block),
// so step s+1's first tiles can start while step s's tail is still running.
//
// Baselines execute the *same* tape with the classic model: one launch per op-phase, and the same
// launches captured in a CUDA graph.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 kvm.cu -o kvm
// Run:   ./kvm [--steps 1000] [--B 128] [--D 128] [--H 256] [--O 64] [--nb 8] [--lr 0.05] [--ref-steps 100]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

enum { OP_GEMM = 0, OP_SGD = 1 };
enum { F_TA = 1, F_TB = 2, F_RELU = 4, F_MSE = 8, F_RELUMASK = 16, F_SGD = 32 };

struct Buf { float* p; int ld; };
__constant__ Buf g_buf[64];

struct Instr {                 // 80 bytes
  int op, flags;
  int a, b, c, aux;            // buffer ids
  int m0, n0, K;
  float scale;
  int w0c, w0v, w1c, w1v;      // wait: counter id (-1 = none) >= value
  int s0, s1;                  // signal: counter ids to increment (-1 = none)
  int loss_idx, pad0, pad1;
};

constexpr int TILE = 32, NT = 128;

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

// ---- pipelined tile GEMM: 4-stage cp.async ring of K-chunks in dynamic smem
constexpr int KC = 64, STG = 4, LDK = KC + 4, LDM = TILE + 4, SLOT = KC * LDM;   // floats per A or B slot
constexpr size_t DSMEM = (size_t)STG * 2 * SLOT * sizeof(float);
extern __shared__ __align__(128) float dsm[];
__shared__ unsigned long long g_stamp[2];
__device__ __forceinline__ unsigned long long gtime();
__device__ bool g_prof_on = false;

__device__ __forceinline__ void cp16(float* dst, const float* src) {
  unsigned d = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait_stg() { asm volatile("cp.async.wait_group %0;" :: "n"(STG - 2) : "memory"); }

template <bool TA, bool TB>
__device__ __forceinline__ void load_chunk(float* sA, float* sB, const Buf& A, const Buf& B, int m0, int n0, int k0) {
  for (int i = threadIdx.x; i < 512; i += NT) {
    if (TA) { int r = i >> 3, c = (i & 7) * 4; cp16(sA + r * LDM + c, A.p + (size_t)(k0 + r) * A.ld + m0 + c); }
    else    { int r = i >> 4, c = (i & 15) * 4; cp16(sA + r * LDK + c, A.p + (size_t)(m0 + r) * A.ld + k0 + c); }
    if (TB) { int r = i >> 4, c = (i & 15) * 4; cp16(sB + r * LDK + c, B.p + (size_t)(n0 + r) * B.ld + k0 + c); }
    else    { int r = i >> 3, c = (i & 7) * 4; cp16(sB + r * LDM + c, B.p + (size_t)(k0 + r) * B.ld + n0 + c); }
  }
}

template <bool TA, bool TB>
__device__ void gemm_acc(const Buf A, const Buf B, int m0, int n0, int K, float* sm) {
  int warp = threadIdx.x >> 5, wm = warp >> 1, wn = warp & 1;
  using AL = typename std::conditional<TA, wmma::col_major, wmma::row_major>::type;
  using BL = typename std::conditional<TB, wmma::col_major, wmma::row_major>::type;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, AL> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, BL> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
  wmma::fill_fragment(acc, 0.f);
  int nch = K / KC;
  for (int s = 0; s < STG - 1; s++) {
    if (s < nch) load_chunk<TA, TB>(dsm + s * 2 * SLOT, dsm + s * 2 * SLOT + SLOT, A, B, m0, n0, s * KC);
    cp_commit();
  }
  for (int c = 0; c < nch; c++) {
    cp_wait_stg();
    __syncthreads();
    if (c == 0 && threadIdx.x == 0 && g_prof_on) g_stamp[0] = gtime();
    int nx = c + STG - 1;
    if (nx < nch) { float* p = dsm + (nx % STG) * 2 * SLOT; load_chunk<TA, TB>(p, p + SLOT, A, B, m0, n0, nx * KC); }
    cp_commit();
    const float* sA = dsm + (c % STG) * 2 * SLOT; const float* sB = sA + SLOT;
#pragma unroll
    for (int kk = 0; kk < KC; kk += 8) {
      wmma::load_matrix_sync(fa, TA ? sA + kk * LDM + wm * 16 : sA + wm * 16 * LDK + kk, TA ? LDM : LDK);
      wmma::load_matrix_sync(fb, TB ? sB + wn * 16 * LDK + kk : sB + kk * LDM + wn * 16, TB ? LDK : LDM);
#pragma unroll
      for (int i = 0; i < fa.num_elements; i++) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
#pragma unroll
      for (int i = 0; i < fb.num_elements; i++) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
      wmma::mma_sync(acc, fa, fb, acc);
    }
  }
  cp_wait_stg(); asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  if (threadIdx.x == 0 && g_prof_on) g_stamp[1] = gtime();   // all smem ring reads done before next instruction reuses it
  wmma::store_matrix_sync(sm + wm * 16 * TILE + wn * 16, acc, TILE, wmma::mem_row_major);
}

// Executes one instruction. All NT threads. Caller handles dependencies. Ends with __syncthreads-safe state.
__device__ void exec_instr(const Instr& in, float* sm, float* loss) {
  __shared__ float red[NT / 32];
  if (in.op == OP_GEMM) {
    Buf A = g_buf[in.a], B = g_buf[in.b], C = g_buf[in.c];
    int f = in.flags;
    if (f & F_TA) { if (f & F_TB) gemm_acc<true, true>(A, B, in.m0, in.n0, in.K, sm); else gemm_acc<true, false>(A, B, in.m0, in.n0, in.K, sm); }
    else          { if (f & F_TB) gemm_acc<false, true>(A, B, in.m0, in.n0, in.K, sm); else gemm_acc<false, false>(A, B, in.m0, in.n0, in.K, sm); }
    __syncthreads();
    float lsum = 0.f;
    Buf X = g_buf[in.aux >= 0 ? in.aux : 0];
    // batched float4 epilogue: issue all global loads first, then compute + store
    float4 v[2], xa[2], cu[2]; size_t gi[2], xi[2];
#pragma unroll
    for (int it = 0; it < 2; it++) {
      int e4 = threadIdx.x + it * NT, r = e4 >> 3, c = (e4 & 7) * 4;
      v[it] = *reinterpret_cast<const float4*>(sm + r * TILE + c);
      gi[it] = (size_t)(in.m0 + r) * C.ld + in.n0 + c;
      xi[it] = (size_t)(in.m0 + r) * X.ld + in.n0 + c;
      if (f & (F_RELUMASK | F_MSE)) xa[it] = __ldcg(reinterpret_cast<const float4*>(X.p + xi[it]));
      if (f & F_SGD) cu[it] = __ldcg(reinterpret_cast<const float4*>(C.p + gi[it]));
    }
#pragma unroll
    for (int it = 0; it < 2; it++) {
      float* q = &v[it].x; const float* x = &xa[it].x; const float* cc = &cu[it].x;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float t = q[j];
        if (f & F_SGD) t = cc[j] - in.scale * t;
        if (f & F_RELU) t = fmaxf(t, 0.f);
        if (f & F_RELUMASK) t = (x[j] > 0.f) ? t : 0.f;
        if (f & F_MSE) { float d = t - x[j]; lsum += d * d * in.scale * 0.5f; t = d * in.scale; }
        q[j] = t;
      }
      *reinterpret_cast<float4*>(C.p + gi[it]) = v[it];
    }
    if (f & F_MSE) {
      for (int o = 16; o; o >>= 1) lsum += __shfl_xor_sync(0xffffffff, lsum, o);
      if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lsum;
      __syncthreads();
      if (threadIdx.x == 0) { float t = 0; for (int i = 0; i < NT / 32; i++) t += red[i]; atomicAdd(loss + in.loss_idx, t); }
    }
  } else {  // OP_SGD: W(c) -= scale * G(a)
    Buf W = g_buf[in.c], G = g_buf[in.a];
    for (int e = threadIdx.x; e < TILE * TILE; e += NT) {
      int r = e / TILE, c = e % TILE;
      float* wp = W.p + (size_t)(in.m0 + r) * W.ld + in.n0 + c;
      *wp = __ldcg(wp) - in.scale * __ldcg(G.p + (size_t)(in.m0 + r) * G.ld + in.n0 + c);
    }
  }
}

// ---- the VM: persistent, dynamic tape claim, counter-based deps
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

__global__ void __launch_bounds__(NT) vm_kernel(const Instr* tape, int n, int* head, int* ctr, float* loss, unsigned long long* prof) {
  __shared__ float sm[TILE * TILE];
  __shared__ int s_idx;
  for (;;) {
    if (threadIdx.x == 0) s_idx = atomicAdd(head, 1);
    __syncthreads();
    int idx = s_idx;
    if (idx >= n) return;
    Instr in = tape[idx];
    unsigned long long t0 = 0, t1 = 0, t2 = 0;
    if (prof) t0 = gtime();
    if (threadIdx.x == 0) {
      if (in.w0c >= 0) while (ld_acquire(ctr + in.w0c) < in.w0v) __nanosleep(32);
      if (in.w1c >= 0) while (ld_acquire(ctr + in.w1c) < in.w1v) __nanosleep(32);
      __threadfence();
    }
    __syncthreads();
    if (prof) t1 = gtime();
    exec_instr(in, sm, loss);
    __syncthreads();
    if (prof) t2 = gtime();
    if (threadIdx.x == 0) {
      __threadfence();   // cumulative over the block's writes via the bar.sync above
      if (in.s0 >= 0) atomicAdd(ctr + in.s0, 1);
      if (in.s1 >= 0) atomicAdd(ctr + in.s1, 1);
      if (prof) { prof[idx * 6] = t0; prof[idx * 6 + 1] = t1; prof[idx * 6 + 2] = t2; prof[idx * 6 + 3] = gtime(); prof[idx * 6 + 4] = g_stamp[0]; prof[idx * 6 + 5] = g_stamp[1]; }
    }
  }
}

// ---- baseline: one launch per phase, block i runs tape[off+i], no waits
__global__ void __launch_bounds__(NT) phase_kernel(const Instr* tape, int off, float* loss) {
  __shared__ float sm[TILE * TILE];
  exec_instr(tape[off + blockIdx.x], sm, loss);
}

// ================= host =================
struct Rng { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 40) & 0xFFFFFF) / 16777216.f * 2.f - 1.f; } };

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  int steps = 1000, B = 128, D = 128, Hd = 256, O = 64, NB = 8, ref_steps = 100;
  float lr = 0.05f; bool prof_on = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return atof(argv[++i]); };
    if (a == "--steps") steps = nx(); else if (a == "--B") B = nx(); else if (a == "--D") D = nx();
    else if (a == "--H") Hd = nx(); else if (a == "--O") O = nx(); else if (a == "--nb") NB = nx();
    else if (a == "--prof") prof_on = true; else if (a == "--lr") lr = nx(); else if (a == "--ref-steps") ref_steps = nx();
  }
  if (B % 64 || D % 64 || Hd % 64 || O % 64) { fprintf(stderr, "dims must be multiples of 64\n"); return 1; }
  int nRB = B / 32, nHC = Hd / 32, nOC = O / 32, nDC = D / 32;

  // ---- data: teacher-generated regression targets
  Rng rng{12345};
  std::vector<float> W1(D * Hd), W2(Hd * O), Wt1(D * Hd), Wt2(Hd * O);
  for (auto& v : W1) v = rng.next() * sqrtf(3.f / D);
  for (auto& v : W2) v = rng.next() * sqrtf(3.f / Hd);
  for (auto& v : Wt1) v = rng.next() * sqrtf(3.f / D);
  for (auto& v : Wt2) v = rng.next() * sqrtf(3.f / Hd);
  std::vector<std::vector<float>> X(NB, std::vector<float>(B * D)), T(NB, std::vector<float>(B * O));
  for (int k = 0; k < NB; k++) {
    for (auto& v : X[k]) v = rng.next();
    std::vector<float> h(B * Hd, 0.f);
    for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) h[i * Hd + j] += x * Wt1[kk * Hd + j]; }
    for (auto& v : h) v = v > 0 ? v : 0;
    std::fill(T[k].begin(), T[k].end(), 0.f);
    for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) T[k][i * O + j] += x * Wt2[kk * O + j]; }
  }

  // ---- CPU reference (fp32)
  std::vector<float> ref_loss;
  std::vector<float> rW1 = W1, rW2 = W2;
  int rs = std::min(steps, ref_steps);
  {
    double t0 = now_ms();
    std::vector<float> h(B * Hd), dY(B * O), dH(B * Hd), g1(D * Hd), g2(Hd * O);
    for (int s = 0; s < rs; s++) {
      int k = s % NB;
      std::fill(h.begin(), h.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) h[i * Hd + j] += x * rW1[kk * Hd + j]; }
      for (auto& v : h) v = v > 0 ? v : 0;
      std::fill(dY.begin(), dY.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) dY[i * O + j] += x * rW2[kk * O + j]; }
      double L = 0;
      for (int i = 0; i < B * O; i++) { float d = dY[i] - T[k][i]; L += (double)d * d; dY[i] = d * 2.f / (B * O); }
      ref_loss.push_back((float)(L / (B * O)));
      std::fill(g2.begin(), g2.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) g2[kk * O + j] += x * dY[i * O + j]; }
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float a = 0; for (int j = 0; j < O; j++) a += dY[i * O + j] * rW2[kk * O + j]; dH[i * Hd + kk] = h[i * Hd + kk] > 0 ? a : 0; }
      std::fill(g1.begin(), g1.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) g1[kk * Hd + j] += x * dH[i * Hd + j]; }
      for (int i = 0; i < D * Hd; i++) rW1[i] -= lr * g1[i];
      for (int i = 0; i < Hd * O; i++) rW2[i] -= lr * g2[i];
    }
    printf("cpu ref: %d steps in %.0f ms\n", rs, now_ms() - t0);
  }

  // ---- device buffers
  enum { bX = 0, bT = 0 + 0 };  // placeholders (ids computed below)
  int idX = 0, idT = NB, idW1 = 2 * NB, idW2 = idW1 + 1, idH1 = idW1 + 2, idDY = idW1 + 3, idDH = idW1 + 4, idG1 = idW1 + 5, idG2 = idW1 + 6;
  int nbuf = idG2 + 1;
  if (nbuf > 64) { fprintf(stderr, "too many buffers\n"); return 1; }
  std::vector<Buf> hb(nbuf);
  auto dmalloc = [&](size_t n) { float* p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; };
  for (int k = 0; k < NB; k++) {
    hb[idX + k] = {dmalloc(B * D), D}; CK(cudaMemcpy(hb[idX + k].p, X[k].data(), B * D * 4, cudaMemcpyHostToDevice));
    hb[idT + k] = {dmalloc(B * O), O}; CK(cudaMemcpy(hb[idT + k].p, T[k].data(), B * O * 4, cudaMemcpyHostToDevice));
  }
  hb[idW1] = {dmalloc(D * Hd), Hd}; hb[idW2] = {dmalloc(Hd * O), O};
  hb[idH1] = {dmalloc(B * Hd), Hd}; hb[idDY] = {dmalloc(B * O), O}; hb[idDH] = {dmalloc(B * Hd), Hd};
  hb[idG1] = {dmalloc(D * Hd), Hd}; hb[idG2] = {dmalloc(Hd * O), O};
  CK(cudaMemcpyToSymbol(g_buf, hb.data(), nbuf * sizeof(Buf)));

  // ---- build tape for the whole run
  constexpr int NPH = 5;  // H1, Y, DH1, DW2(+SGD), DW1(+SGD)
  enum { P_H1, P_Y, P_DH1, P_DW2, P_DW1 };
  int CPS = NPH * (nRB + 1);   // counters per step
  auto ctr = [&](int s, int ph, int rb) { return s * CPS + ph * (nRB + 1) + (rb < 0 ? nRB : rb); };
  std::vector<Instr> tape;
  std::vector<int> phase_off, phase_n;   // for baselines
  auto mk = [&]() { Instr i; memset(&i, 0, sizeof i); i.aux = -1; i.w0c = i.w1c = i.s0 = i.s1 = -1; return i; };
  for (int s = 0; s < steps; s++) {
    int k = s % NB;
    auto begin = [&]() { phase_off.push_back((int)tape.size()); };
    auto end = [&]() { phase_n.push_back((int)tape.size() - phase_off.back()); };
    // H1 = relu(X W1)   (W1 updated by DW1(s-1); H1 buffer read by DW2(s-1))
    begin();
    for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_RELU; i.a = idX + k; i.b = idW1; i.c = idH1; i.m0 = rb * 32; i.n0 = cn * 32; i.K = D;
      if (s > 0) { i.w0c = ctr(s - 1, P_DW1, -1); i.w0v = nDC * nHC; i.w1c = ctr(s - 1, P_DW2, -1); i.w1v = nHC * nOC; }
      i.s0 = ctr(s, P_H1, rb); i.s1 = ctr(s, P_H1, -1); tape.push_back(i);
    }
    end();
    // dY = 2/(BO) (H1 W2 - T); loss   (W2 updated by DW2(s-1))
    begin();
    for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nOC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_MSE; i.a = idH1; i.b = idW2; i.c = idDY; i.aux = idT + k; i.m0 = rb * 32; i.n0 = cn * 32; i.K = Hd;
      i.scale = 2.f / (B * O); i.loss_idx = s;
      i.w0c = ctr(s, P_H1, rb); i.w0v = nHC;
      if (s > 0) { i.w1c = ctr(s - 1, P_DW2, -1); i.w1v = nHC * nOC; }
      i.s0 = ctr(s, P_Y, rb); i.s1 = ctr(s, P_Y, -1); tape.push_back(i);
    }
    end();
    // dH1 = (dY W2^T) * (H1>0)   (reads old W2, so must precede DW2's in-place SGD)
    begin();
    for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TB | F_RELUMASK; i.a = idDY; i.b = idW2; i.c = idDH; i.aux = idH1; i.m0 = rb * 32; i.n0 = cn * 32; i.K = O;
      i.w0c = ctr(s, P_Y, rb); i.w0v = nOC; i.s1 = ctr(s, P_DH1, -1); tape.push_back(i);
    }
    end();
    // W2 -= lr * H1^T dY   (fused SGD epilogue; waits for all dH1 tiles that read W2)
    begin();
    for (int m = 0; m < nHC; m++) for (int n = 0; n < nOC; n++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idH1; i.b = idDY; i.c = idW2; i.m0 = m * 32; i.n0 = n * 32; i.K = B; i.scale = lr;
      i.w0c = ctr(s, P_Y, -1); i.w0v = nRB * nOC; i.w1c = ctr(s, P_DH1, -1); i.w1v = nRB * nHC;
      i.s1 = ctr(s, P_DW2, -1); tape.push_back(i);
    }
    end();
    // W1 -= lr * X^T dH1
    begin();
    for (int m = 0; m < nDC; m++) for (int n = 0; n < nHC; n++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idX + k; i.b = idDH; i.c = idW1; i.m0 = m * 32; i.n0 = n * 32; i.K = B; i.scale = lr;
      i.w0c = ctr(s, P_DH1, -1); i.w0v = nRB * nHC; i.s1 = ctr(s, P_DW1, -1); tape.push_back(i);
    }
    end();
  }
  int ntape = (int)tape.size(), nctr = steps * CPS;
  printf("model B=%d D=%d H=%d O=%d  steps=%d  tape=%d instrs (%.1f MB)  phases=%zu\n", B, D, Hd, O, steps, ntape, ntape * sizeof(Instr) / 1e6, phase_off.size());

  Instr* dtape; int *dhead, *dctr; float* dloss;
  CK(cudaMalloc(&dtape, ntape * sizeof(Instr))); CK(cudaMemcpy(dtape, tape.data(), ntape * sizeof(Instr), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dhead, 4)); CK(cudaMalloc(&dctr, nctr * 4)); CK(cudaMalloc(&dloss, steps * 4));

  CK(cudaFuncSetAttribute(vm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  CK(cudaFuncSetAttribute(phase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm_kernel, NT, DSMEM));
  int grid = prop.multiProcessorCount * occ;
  printf("%s  SMs=%d occ=%d  vm grid=%d\n", prop.name, prop.multiProcessorCount, occ, grid);

  auto reset = [&]() {
    CK(cudaMemcpy(hb[idW1].p, W1.data(), W1.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(hb[idW2].p, W2.data(), W2.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dhead, 0, 4)); CK(cudaMemset(dctr, 0, nctr * 4)); CK(cudaMemset(dloss, 0, steps * 4));
    CK(cudaDeviceSynchronize());
  };
  auto launch_phases = [&](cudaStream_t st) {
    for (size_t p = 0; p < phase_off.size(); p++) phase_kernel<<<phase_n[p], NT, DSMEM, st>>>(dtape, phase_off[p], dloss);
  };
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto timed = [&](auto fn) { reset(); cudaEventRecord(e0); fn(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError()); float ms; cudaEventElapsedTime(&ms, e0, e1); return (double)ms; };
  auto fetch = [&](std::vector<float>& L, std::vector<float>& w1, std::vector<float>& w2) {
    L.resize(steps); w1.resize(W1.size()); w2.resize(W2.size());
    CK(cudaMemcpy(L.data(), dloss, steps * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w1.data(), hb[idW1].p, w1.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(w2.data(), hb[idW2].p, w2.size() * 4, cudaMemcpyDeviceToHost));
  };

  std::vector<float> Lv, Lb, Lg, w1v, w2v, w1b, w2b, w1g, w2g;
  // warmup + measure; take best of 3
  auto run_vm = [&]() { vm_kernel<<<grid, NT, DSMEM>>>(dtape, ntape, dhead, dctr, dloss, nullptr); };
  auto run_launch = [&]() { launch_phases(0); };
  cudaGraph_t graph; cudaGraphExec_t gexec;
  { cudaStream_t cs; cudaStreamCreate(&cs); cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_phases(cs); CK(cudaStreamEndCapture(cs, &graph)); CK(cudaGraphInstantiate(&gexec, graph, 0)); }
  auto run_graph = [&]() { CK(cudaGraphLaunch(gexec, 0)); };

  double bv = 1e30, bb = 1e30, bg = 1e30;
  timed(run_vm); for (int r = 0; r < 3; r++) bv = std::min(bv, timed(run_vm)); fetch(Lv, w1v, w2v);
  timed(run_launch); for (int r = 0; r < 3; r++) bb = std::min(bb, timed(run_launch)); fetch(Lb, w1b, w2b);
  timed(run_graph); for (int r = 0; r < 3; r++) bg = std::min(bg, timed(run_graph)); fetch(Lg, w1g, w2g);

  if (prof_on) {
    unsigned long long* dprof; CK(cudaMalloc(&dprof, (size_t)ntape * 48)); CK(cudaMemset(dprof, 0, (size_t)ntape * 48)); { bool on = true; CK(cudaMemcpyToSymbol(g_prof_on, &on, 1)); }
    reset(); vm_kernel<<<grid, NT, DSMEM>>>(dtape, ntape, dhead, dctr, dloss, dprof); CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> P((size_t)ntape * 6); CK(cudaMemcpy(P.data(), dprof, (size_t)ntape * 48, cudaMemcpyDeviceToHost));
    const char* nm[] = {"H1", "Y", "DH1", "DW2", "DW1"};
    int per = (int)phase_off.size() / steps;
    double fsum[8] = {0}, msum[8] = {0}, esum[8] = {0}; long cnt[8] = {0};
    for (int s = steps / 4; s < steps; s++) for (int p = 0; p < per; p++) {
      int pi = s * per + p;
      for (int j = 0; j < phase_n[pi]; j++) { size_t i = (size_t)(phase_off[pi] + j) * 6;
        fsum[p] += (P[i + 4] - P[i + 1]) / 1e3; msum[p] += (P[i + 5] - P[i + 4]) / 1e3; esum[p] += (P[i + 2] - P[i + 5]) / 1e3; cnt[p]++; }
    }
    unsigned long long tmin = ~0ull, tmax = 0; for (int i = 0; i < ntape; i++) { tmin = std::min(tmin, P[i * 6]); tmax = std::max(tmax, P[i * 6 + 3]); }
    printf("\nprofile (us, mean per instr):  phase  first-chunk  mainloop  epilogue\n");
    for (int p = 0; p < per; p++) printf("  %-4s %9.2f %9.2f %9.2f\n", nm[p], fsum[p] / cnt[p], msum[p] / cnt[p], esum[p] / cnt[p]);
    printf("  span %.1f us total\n", (tmax - tmin) / 1e3);
  }
  printf("\n%-34s %10s %12s %10s\n", "mode", "total ms", "us/step", "speedup");
  printf("%-34s %10.2f %12.2f %10s\n", "launch-per-op (classic)", bb, bb * 1000 / steps, "1.00x");
  printf("%-34s %10.2f %12.2f %9.2fx\n", "CUDA graph (same launches)", bg, bg * 1000 / steps, bb / bg);
  printf("%-34s %10.2f %12.2f %9.2fx\n", "KVM: whole run in ONE kernel", bv, bv * 1000 / steps, bb / bv);

  // ---- correctness
  auto relerr = [](const std::vector<float>& a, const std::vector<float>& b, int n) { double m = 0; for (int i = 0; i < n; i++) m = std::max(m, (double)fabs(a[i] - b[i]) / (fabs(b[i]) + 1e-6)); return m; };
  auto maxabs = [](const std::vector<float>& a, const std::vector<float>& b) { double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, (double)fabs(a[i] - b[i])); return m; };
  printf("\nloss  step0 %.5f  step%d %.5f  (vm)   | cpu: %.5f -> %.5f at step %d\n", Lv[0], steps - 1, Lv[steps - 1], ref_loss[0], ref_loss[rs - 1], rs - 1);
  double e_vm = relerr(Lv, ref_loss, rs), e_b = relerr(Lb, ref_loss, rs), e_g = relerr(Lg, ref_loss, rs);
  printf("max rel loss err vs cpu (first %d steps): vm %.2e  launch %.2e  graph %.2e\n", rs, e_vm, e_b, e_g);
  printf("max |dW| between modes (final weights): vm-launch %.2e  vm-graph %.2e\n", std::max(maxabs(w1v, w1b), maxabs(w2v, w2b)), std::max(maxabs(w1v, w1g), maxabs(w2v, w2g)));
  bool ok = e_vm < 2e-2 && e_b < 2e-2 && e_g < 2e-2 && Lv[steps - 1] < Lv[0];
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}
