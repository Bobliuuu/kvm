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
enum { F_TA = 1, F_TB = 2, F_RELU = 4, F_MSE = 8, F_RELUMASK = 16 };

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

template <bool TA, bool TB>
__device__ void gemm_acc(const Buf A, const Buf B, int m0, int n0, int K, float* sm) {
  int warp = threadIdx.x >> 5, wm = warp >> 1, wn = warp & 1;
  using AL = typename std::conditional<TA, wmma::col_major, wmma::row_major>::type;
  using BL = typename std::conditional<TB, wmma::col_major, wmma::row_major>::type;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, AL> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, BL> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
  wmma::fill_fragment(acc, 0.f);
  int mm = m0 + wm * 16, nn = n0 + wn * 16;
  for (int k = 0; k < K; k += 8) {
    const float* ap = TA ? A.p + (size_t)k * A.ld + mm : A.p + (size_t)mm * A.ld + k;
    const float* bp = TB ? B.p + (size_t)nn * B.ld + k : B.p + (size_t)k * B.ld + nn;
    wmma::load_matrix_sync(fa, ap, A.ld);
    wmma::load_matrix_sync(fb, bp, B.ld);
#pragma unroll
    for (int i = 0; i < fa.num_elements; i++) fa.x[i] = wmma::__float_to_tf32(fa.x[i]);
#pragma unroll
    for (int i = 0; i < fb.num_elements; i++) fb.x[i] = wmma::__float_to_tf32(fb.x[i]);
    wmma::mma_sync(acc, fa, fb, acc);
  }
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
    for (int e = threadIdx.x; e < TILE * TILE; e += NT) {
      int r = e / TILE, c = e % TILE;
      float v = sm[e];
      size_t gi = (size_t)(in.m0 + r) * C.ld + in.n0 + c;
      if (f & F_RELU) v = fmaxf(v, 0.f);
      if (f & F_RELUMASK) v = (X.p[(size_t)(in.m0 + r) * X.ld + in.n0 + c] > 0.f) ? v : 0.f;
      if (f & F_MSE) {
        float d = v - X.p[(size_t)(in.m0 + r) * X.ld + in.n0 + c];
        lsum += d * d * in.scale * 0.5f;
        v = d * in.scale;
      }
      C.p[gi] = v;
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
      W.p[(size_t)(in.m0 + r) * W.ld + in.n0 + c] -= in.scale * G.p[(size_t)(in.m0 + r) * G.ld + in.n0 + c];
    }
  }
}

// ---- the VM: persistent, dynamic tape claim, counter-based deps
__global__ void __launch_bounds__(NT) vm_kernel(const Instr* tape, int n, int* head, int* ctr, float* loss) {
  __shared__ float sm[TILE * TILE];
  __shared__ int s_idx;
  for (;;) {
    if (threadIdx.x == 0) s_idx = atomicAdd(head, 1);
    __syncthreads();
    int idx = s_idx;
    if (idx >= n) return;
    Instr in = tape[idx];
    if (threadIdx.x == 0) {
      if (in.w0c >= 0) while (ld_acquire(ctr + in.w0c) < in.w0v) __nanosleep(32);
      if (in.w1c >= 0) while (ld_acquire(ctr + in.w1c) < in.w1v) __nanosleep(32);
      __threadfence();
    }
    __syncthreads();
    exec_instr(in, sm, loss);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      if (in.s0 >= 0) atomicAdd(ctr + in.s0, 1);
      if (in.s1 >= 0) atomicAdd(ctr + in.s1, 1);
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
  float lr = 0.05f;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return atof(argv[++i]); };
    if (a == "--steps") steps = nx(); else if (a == "--B") B = nx(); else if (a == "--D") D = nx();
    else if (a == "--H") Hd = nx(); else if (a == "--O") O = nx(); else if (a == "--nb") NB = nx();
    else if (a == "--lr") lr = nx(); else if (a == "--ref-steps") ref_steps = nx();
  }
  if (B % 32 || D % 32 || Hd % 32 || O % 32) { fprintf(stderr, "dims must be multiples of 32\n"); return 1; }
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
  constexpr int NPH = 7;  // H1, Y, DW2, DH1, DW1, SGD2, SGD1
  enum { P_H1, P_Y, P_DW2, P_DH1, P_DW1, P_S2, P_S1 };
  int CPS = NPH * (nRB + 1);   // counters per step
  auto ctr = [&](int s, int ph, int rb) { return s * CPS + ph * (nRB + 1) + (rb < 0 ? nRB : rb); };
  std::vector<Instr> tape;
  std::vector<int> phase_off, phase_n;   // for baselines
  auto mk = [&]() { Instr i; memset(&i, 0, sizeof i); i.aux = -1; i.w0c = i.w1c = i.s0 = i.s1 = -1; return i; };
  for (int s = 0; s < steps; s++) {
    int k = s % NB;
    auto begin = [&]() { phase_off.push_back((int)tape.size()); };
    auto end = [&]() { phase_n.push_back((int)tape.size() - phase_off.back()); };
    // H1 = relu(X W1)
    begin();
    for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_RELU; i.a = idX + k; i.b = idW1; i.c = idH1; i.m0 = rb * 32; i.n0 = cn * 32; i.K = D;
      if (s > 0) { i.w0c = ctr(s - 1, P_S1, -1); i.w0v = nDC * nHC; i.w1c = ctr(s - 1, P_DW2, -1); i.w1v = nHC * nOC; }
      i.s0 = ctr(s, P_H1, rb); i.s1 = ctr(s, P_H1, -1); tape.push_back(i);
    }
    end();
    // dY = 2/(BO) (H1 W2 - T); loss
    begin();
    for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nOC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_MSE; i.a = idH1; i.b = idW2; i.c = idDY; i.aux = idT + k; i.m0 = rb * 32; i.n0 = cn * 32; i.K = Hd;
      i.scale = 2.f / (B * O); i.loss_idx = s;
      i.w0c = ctr(s, P_H1, rb); i.w0v = nHC;
      if (s > 0) { i.w1c = ctr(s - 1, P_S2, -1); i.w1v = nHC * nOC; }
      i.s0 = ctr(s, P_Y, rb); i.s1 = ctr(s, P_Y, -1); tape.push_back(i);
    }
    end();
    // dW2 = H1^T dY
    begin();
    for (int m = 0; m < nHC; m++) for (int n = 0; n < nOC; n++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA; i.a = idH1; i.b = idDY; i.c = idG2; i.m0 = m * 32; i.n0 = n * 32; i.K = B;
      i.w0c = ctr(s, P_Y, -1); i.w0v = nRB * nOC; i.s1 = ctr(s, P_DW2, -1); tape.push_back(i);
    }
    end();
    // dH1 = (dY W2^T) * (H1>0)
    begin();
    for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TB | F_RELUMASK; i.a = idDY; i.b = idW2; i.c = idDH; i.aux = idH1; i.m0 = rb * 32; i.n0 = cn * 32; i.K = O;
      i.w0c = ctr(s, P_Y, rb); i.w0v = nOC; i.s1 = ctr(s, P_DH1, -1); tape.push_back(i);
    }
    end();
    // dW1 = X^T dH1
    begin();
    for (int m = 0; m < nDC; m++) for (int n = 0; n < nHC; n++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA; i.a = idX + k; i.b = idDH; i.c = idG1; i.m0 = m * 32; i.n0 = n * 32; i.K = B;
      i.w0c = ctr(s, P_DH1, -1); i.w0v = nRB * nHC; i.s1 = ctr(s, P_DW1, -1); tape.push_back(i);
    }
    end();
    // SGD W2
    begin();
    for (int m = 0; m < nHC; m++) for (int n = 0; n < nOC; n++) {
      Instr i = mk(); i.op = OP_SGD; i.a = idG2; i.c = idW2; i.m0 = m * 32; i.n0 = n * 32; i.scale = lr;
      i.w0c = ctr(s, P_DW2, -1); i.w0v = nHC * nOC; i.w1c = ctr(s, P_DH1, -1); i.w1v = nRB * nHC;
      i.s1 = ctr(s, P_S2, -1); tape.push_back(i);
    }
    end();
    // SGD W1
    begin();
    for (int m = 0; m < nDC; m++) for (int n = 0; n < nHC; n++) {
      Instr i = mk(); i.op = OP_SGD; i.a = idG1; i.c = idW1; i.m0 = m * 32; i.n0 = n * 32; i.scale = lr;
      i.w0c = ctr(s, P_DW1, -1); i.w0v = nDC * nHC; i.s1 = ctr(s, P_S1, -1); tape.push_back(i);
    }
    end();
  }
  int ntape = (int)tape.size(), nctr = steps * CPS;
  printf("model B=%d D=%d H=%d O=%d  steps=%d  tape=%d instrs (%.1f MB)  phases=%zu\n", B, D, Hd, O, steps, ntape, ntape * sizeof(Instr) / 1e6, phase_off.size());

  Instr* dtape; int *dhead, *dctr; float* dloss;
  CK(cudaMalloc(&dtape, ntape * sizeof(Instr))); CK(cudaMemcpy(dtape, tape.data(), ntape * sizeof(Instr), cudaMemcpyHostToDevice));
  CK(cudaMalloc(&dhead, 4)); CK(cudaMalloc(&dctr, nctr * 4)); CK(cudaMalloc(&dloss, steps * 4));

  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm_kernel, NT, 0));
  int grid = prop.multiProcessorCount * occ;
  printf("%s  SMs=%d occ=%d  vm grid=%d\n", prop.name, prop.multiProcessorCount, occ, grid);

  auto reset = [&]() {
    CK(cudaMemcpy(hb[idW1].p, W1.data(), W1.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(hb[idW2].p, W2.data(), W2.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(dhead, 0, 4)); CK(cudaMemset(dctr, 0, nctr * 4)); CK(cudaMemset(dloss, 0, steps * 4));
    CK(cudaDeviceSynchronize());
  };
  auto launch_phases = [&](cudaStream_t st) {
    for (size_t p = 0; p < phase_off.size(); p++) phase_kernel<<<phase_n[p], NT, 0, st>>>(dtape, phase_off[p], dloss);
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
  auto run_vm = [&]() { vm_kernel<<<grid, NT>>>(dtape, ntape, dhead, dctr, dloss); };
  auto run_launch = [&]() { launch_phases(0); };
  cudaGraph_t graph; cudaGraphExec_t gexec;
  { cudaStream_t cs; cudaStreamCreate(&cs); cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_phases(cs); CK(cudaStreamEndCapture(cs, &graph)); CK(cudaGraphInstantiate(&gexec, graph, 0)); }
  auto run_graph = [&]() { CK(cudaGraphLaunch(gexec, 0)); };

  double bv = 1e30, bb = 1e30, bg = 1e30;
  timed(run_vm); for (int r = 0; r < 3; r++) bv = std::min(bv, timed(run_vm)); fetch(Lv, w1v, w2v);
  timed(run_launch); for (int r = 0; r < 3; r++) bb = std::min(bb, timed(run_launch)); fetch(Lb, w1b, w2b);
  timed(run_graph); for (int r = 0; r < 3; r++) bg = std::min(bg, timed(run_graph)); fetch(Lg, w1g, w2g);

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
