// kvm: "kittens"-style GPU virtual machine, v2 prototype (sm_80).
//
// ONE persistent kernel interprets an instruction tape. The tape encodes an ENTIRE training run
// (every step: fwd, loss, bwd, SGD) of a 2-layer MLP. No kernel boundaries between ops or steps:
// instructions synchronise through monotonic global counters (fine-grained, per row-block),
// so step s+1's first tiles can start while step s's tail is still running.
//
// Baselines execute the *same* tape with the classic model: one launch per op-phase, and the same
// launches captured in a CUDA graph. A fifth mode runs the same tape through a dependency-driven
// ready queue (host-lowered DAG, device-side successor release) instead of claim-and-wait.
//
// Build: nvcc -O3 -arch=sm_80 -std=c++17 kvm.cu -o kvm
// Run:   ./kvm [--steps 1000] [--B 128] [--D 128] [--H 256] [--O 64] [--nb 8] [--lr 0.05] [--ref-steps 100]
//             [--runs R] [--reps N] [--modes seq,seqgraph,batched,claim,queue] [--watchdog-ms 5000] [--quiet] [--inject-fault]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <functional>
#include <chrono>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

enum { OP_GEMM = 0, OP_SGD = 1 };
enum { F_TA = 1, F_TB = 2, F_RELU = 4, F_MSE = 8, F_RELUMASK = 16, F_SGD = 32 };

struct Buf { float* p; int ld; };
__constant__ Buf g_buf[256];


struct Instr {                 // 80 bytes
  int op, flags;
  int a, b, c, aux;            // buffer ids
  int m0, n0, K;
  float scale;
  int w0c, w0v, w1c, w1v;      // wait: counter id (-1 = none) >= value
  int s0, s1;                  // signal: counter ids to increment (-1 = none)
  int loss_idx, xflags, pad1;   // xflags: 1/2 = w0/w1 wait on previous step, 4 = a is X(step%NB), 8 = aux is T(step%NB)
};

struct KP { int SC, NB, perStep; };   // SC = counters per step (all runs), NB = data batches, perStep = template length
__constant__ KP g_kp;

// Loop opcode: the tape stores ONE step as a template; step s instantiates it on the device.
// Host and device share this one definition, so the queue's host-lowered DAG cannot drift from what the VM executes.
__host__ __device__ __forceinline__ Instr instantiate(Instr in, int s, int SC, int NB) {
  int xf = in.xflags;
  if (xf & 1) { in.w0c = s ? in.w0c + (s - 1) * SC : -1; } else if (in.w0c >= 0) in.w0c += s * SC;
  if (xf & 2) { in.w1c = s ? in.w1c + (s - 1) * SC : -1; } else if (in.w1c >= 0) in.w1c += s * SC;
  if (in.s0 >= 0) in.s0 += s * SC;
  if (in.s1 >= 0) in.s1 += s * SC;
  if (xf & 4) in.a += s % NB;
  if (xf & 8) in.aux += s % NB;
  in.loss_idx += s;
  return in;
}
__device__ __forceinline__ Instr inst_at(const Instr* tape, int t, int s) { return instantiate(tape[t], s, g_kp.SC, g_kp.NB); }

// ---- device watchdog. Every spin is bounded; the first block to give up records why in host-mapped memory and every
// other spinning block then exits, so a dependency bug ends the kernel with a diagnosis instead of hanging the GPU.
struct Diag {
  int tripped;                       // 0 = healthy, 1 = a block gave up (fields below describe the first one)
  int block, idx, ctr, want, seen;   // claim VM: instruction idx waiting on counter ctr (needs >= want, saw seen)
  int head, tail, done, n;           // queue VM: queue state when the idle block gave up
  unsigned long long limit_ns;       // set by the host before launch
};

#ifndef KTILE
#define KTILE 64
#endif
constexpr int TILE = KTILE, NT = 128, WT = TILE / 2, FR = WT / 16, VPT = TILE / 4;   // warp block, frags/dim, 16B vectors per TILE-wide row

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}

// ---- pipelined 64x64 tile GEMM: STG-stage cp.async ring of K-chunks in dynamic smem.
// 4 warps in a 2x2 grid, each owning a 32x32 output block (2x2 wmma 16x16x8 TF32 fragments).
#ifndef KKC
#define KKC 32
#endif
#ifndef KSTG
#define KSTG 3
#endif
constexpr int KC = KKC, STG = KSTG, VPK = KC / 4, LDK = KC + 4, LDM = TILE + 4, SLOT = (TILE * LDK > KC * LDM ? TILE * LDK : KC * LDM), NV = TILE * KC / 4 / NT;
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
#pragma unroll
  for (int j = 0; j < NV; j++) {
    int i = threadIdx.x + j * NT;      // TILE*KC/4 16-byte vectors per operand chunk
    if (TA) { int r = i / VPT, c = (i % VPT) * 4; cp16(sA + r * LDM + c, A.p + (size_t)(k0 + r) * A.ld + m0 + c); }
    else    { int r = i / VPK, c = (i % VPK) * 4;    cp16(sA + r * LDK + c, A.p + (size_t)(m0 + r) * A.ld + k0 + c); }
    if (TB) { int r = i / VPK, c = (i % VPK) * 4;    cp16(sB + r * LDK + c, B.p + (size_t)(n0 + r) * B.ld + k0 + c); }
    else    { int r = i / VPT, c = (i % VPT) * 4; cp16(sB + r * LDM + c, B.p + (size_t)(k0 + r) * B.ld + n0 + c); }
  }
}

template <bool TA, bool TB>
__device__ void gemm_acc(const Buf A, const Buf B, int m0, int n0, int K, float* sm) {
  int warp = threadIdx.x >> 5, wm = warp >> 1, wn = warp & 1;
  using AL = typename std::conditional<TA, wmma::col_major, wmma::row_major>::type;
  using BL = typename std::conditional<TB, wmma::col_major, wmma::row_major>::type;
  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, AL> fa[FR];
  wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, BL> fb[FR];
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[FR][FR];
#pragma unroll
  for (int i = 0; i < FR; i++)
#pragma unroll
    for (int j = 0; j < FR; j++) wmma::fill_fragment(acc[i][j], 0.f);
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
#pragma unroll
      for (int i = 0; i < FR; i++) {
        wmma::load_matrix_sync(fa[i], TA ? sA + kk * LDM + wm * WT + i * 16 : sA + (wm * WT + i * 16) * LDK + kk, TA ? LDM : LDK);
#pragma unroll
        for (int e = 0; e < fa[i].num_elements; e++) fa[i].x[e] = wmma::__float_to_tf32(fa[i].x[e]);
      }
#pragma unroll
      for (int j = 0; j < FR; j++) {
        wmma::load_matrix_sync(fb[j], TB ? sB + (wn * WT + j * 16) * LDK + kk : sB + kk * LDM + wn * WT + j * 16, TB ? LDK : LDM);
#pragma unroll
        for (int e = 0; e < fb[j].num_elements; e++) fb[j].x[e] = wmma::__float_to_tf32(fb[j].x[e]);
      }
#pragma unroll
      for (int i = 0; i < FR; i++)
#pragma unroll
        for (int j = 0; j < FR; j++) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }
  cp_wait_stg(); asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();   // ring reads done before the next instruction reuses it
  if (threadIdx.x == 0 && g_prof_on) g_stamp[1] = gtime();
#pragma unroll
  for (int i = 0; i < FR; i++)
#pragma unroll
    for (int j = 0; j < FR; j++) wmma::store_matrix_sync(sm + (wm * WT + i * 16) * TILE + wn * WT + j * 16, acc[i][j], TILE, wmma::mem_row_major);
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
    constexpr int EPI = TILE * TILE / 4 / NT;
    float4 v[EPI], xa[EPI], cu[EPI]; size_t gi[EPI], xi[EPI];
#pragma unroll
    for (int it = 0; it < EPI; it++) {
      int e4 = threadIdx.x + it * NT, r = e4 / VPT, c = (e4 % VPT) * 4;
      v[it] = *reinterpret_cast<const float4*>(sm + r * TILE + c);
      gi[it] = (size_t)(in.m0 + r) * C.ld + in.n0 + c;
      xi[it] = (size_t)(in.m0 + r) * X.ld + in.n0 + c;
      if (f & (F_RELUMASK | F_MSE)) xa[it] = __ldcg(reinterpret_cast<const float4*>(X.p + xi[it]));
      if (f & F_SGD) cu[it] = __ldcg(reinterpret_cast<const float4*>(C.p + gi[it]));
    }
#pragma unroll
    for (int it = 0; it < EPI; it++) {
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

// Records the first give-up; returns true if this block is the first. Callers then leave the kernel.
__device__ bool watchdog_trip(Diag* d) { return atomicCAS(&d->tripped, 0, 1) == 0; }
__device__ __forceinline__ bool watchdog_tripped(const Diag* d) { return ld_acquire(&d->tripped) != 0; }

// Spins until *(ctr+c) >= want. Returns false (with *seen set) if the watchdog expired or another block already gave up.
__device__ bool wait_counter(const int* ctr, int c, int want, const Diag* d, int* seen) {
  int cur = ld_acquire(ctr + c);
  if (cur >= want) return true;
  unsigned long long t0 = gtime();
  for (unsigned spin = 1; cur < want; spin++) {
    __nanosleep(32);
    cur = ld_acquire(ctr + c);
    if ((spin & 255) == 0 && (watchdog_tripped(d) || gtime() - t0 > d->limit_ns)) { *seen = cur; return cur >= want; }
  }
  return true;
}

__global__ void __launch_bounds__(NT) vm_kernel(const Instr* tape, int n, int* head, int* ctr, float* loss, unsigned long long* prof, Diag* diag) {   // n = steps * perStep
  __shared__ __align__(128) float sm[TILE * TILE];
  __shared__ int s_idx;
  for (;;) {
    if (threadIdx.x == 0) s_idx = atomicAdd(head, 1);
    __syncthreads();
    int idx = s_idx;
    if (idx >= n) return;
    int stp = idx / g_kp.perStep;
    Instr in = inst_at(tape, idx - stp * g_kp.perStep, stp);
    unsigned long long t0 = 0, t1 = 0, t2 = 0;
    if (prof) t0 = gtime();
    if (threadIdx.x == 0) {
      int seen = 0, bad = -1;
      if (in.w0c >= 0 && !wait_counter(ctr, in.w0c, in.w0v, diag, &seen)) bad = 0;
      else if (in.w1c >= 0 && !wait_counter(ctr, in.w1c, in.w1v, diag, &seen)) bad = 1;
      if (bad >= 0) {
        if (watchdog_trip(diag)) { diag->block = blockIdx.x; diag->idx = idx; diag->ctr = bad ? in.w1c : in.w0c; diag->want = bad ? in.w1v : in.w0v; diag->seen = seen; diag->n = n; }
        s_idx = -1;
      }
      __threadfence();
    }
    __syncthreads();
    if (s_idx < 0) return;
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

// ---- the ready-queue VM: the host lowers the counter dependencies of the whole tape into a CSR successor DAG. A block takes
// a ticket (atomicAdd on head), waits for that queue slot to be published, runs the task, then the block's threads
// decrement the remaining-predecessor count of the task's successors in parallel and push each one that reaches zero.
// Nothing waits on a counter. Capacity is n slots and every task is pushed exactly once, so the queue never wraps or fills,
// and a ticket >= n means all work is handed out. A slot is reserved (tail++) before it is published (ready=1), so a
// ticket holder spins on its own slot only: a stalled producer delays one consumer and never skips or duplicates work.
// Progress: every slot's predecessors sit in lower slots, so the lowest unpublished slot's holder waits only on tasks
// that running blocks will finish; like the claim VM this needs all blocks resident (grid = SMs * occupancy).
struct QueueDev {                            // device pointers of one queue instance
  const int *offsets, *successors;           // CSR successor lists
  int *remaining;                            // unfinished predecessors per task
  int *task, *ready;                         // queue slots: task id, published flag
  int *head, *tail, *done;                   // ticket counter, push cursor, finished-task count (diagnostics)
};

__global__ void __launch_bounds__(NT) queue_kernel(const Instr* tape, int n, QueueDev q, float* loss, Diag* diag) {
  __shared__ __align__(128) float sm[TILE * TILE];
  __shared__ int s_task;                     // >= 0 task id, -1 all handed out, -2 watchdog
  for (;;) {
    if (threadIdx.x == 0) {
      int ticket = atomicAdd(q.head, 1);
      s_task = -1;
      if (ticket < n) {
        const int* flag = q.ready + ticket;
        int spin = 0; unsigned long long t0 = gtime();
        while (!ld_acquire(flag)) {
          __nanosleep(32);
          if ((++spin & 255) == 0 && (watchdog_tripped(diag) || gtime() - t0 > diag->limit_ns)) { s_task = -2; break; }
        }
        if (s_task != -2) s_task = __ldcg(q.task + ticket);
        else if (watchdog_trip(diag)) { diag->block = blockIdx.x; diag->idx = ticket; diag->head = ld_acquire(q.head); diag->tail = ld_acquire(q.tail); diag->done = ld_acquire(q.done); diag->n = n; }
      }
    }
    __syncthreads();
    int task = s_task;
    if (task < 0) return;
    int stp = task / g_kp.perStep;
    exec_instr(inst_at(tape, task - stp * g_kp.perStep, stp), sm, loss);
    __syncthreads();
    __threadfence();   // every thread's writes (via the barrier above) are visible before any successor can be released
    for (int e = q.offsets[task] + threadIdx.x; e < q.offsets[task + 1]; e += NT) {
      int succ = q.successors[e];
      if (atomicSub(q.remaining + succ, 1) == 1) {
        __threadfence();   // last releaser: also publish the other predecessors' writes it has observed
        int slot = atomicAdd(q.tail, 1);
        q.task[slot] = succ;
        __threadfence();
        atomicExch(q.ready + slot, 1);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(q.done, 1);
  }
}

// ---- baseline: one launch per phase, block i runs tape[off+i], no waits
__global__ void __launch_bounds__(NT) phase_kernel(const Instr* tape, int off, int step, float* loss) {
  __shared__ __align__(128) float sm[TILE * TILE];
  exec_instr(inst_at(tape, off + blockIdx.x, step), sm, loss);
}

// ================= host =================
struct Rng { uint64_t s; float next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return ((s >> 40) & 0xFFFFFF) / 16777216.f * 2.f - 1.f; } };

// ---- logging: --quiet keeps errors only. Everything else is on by default so a failure can be diagnosed from the log alone.
static bool g_verbose = true;
#define LOGI(...) do { if (g_verbose) { printf(__VA_ARGS__); } } while (0)

// ---- ready-queue lowering. Instantiates every task exactly as the device does (`instantiate`), turns each counter wait into
// edges from all producers of that counter, and checks the result is a DAG before a persistent kernel is launched.
// Only full-barrier thresholds (wait value 0 or the counter's total producer count) have a static meaning independent of
// completion order; anything else is rejected rather than silently lowered to a different dependency.
struct ReadyGraph { std::vector<int> offsets, successors, indegree, initial; };

static ReadyGraph lower_to_dag(const std::vector<Instr>& tmpl, int steps, int SC, int NB) {
  const int perStep = (int)tmpl.size(), n = perStep * steps;
  std::vector<Instr> task(n);
  std::vector<std::vector<int>> producers((size_t)steps * SC);
  for (int id = 0; id < n; id++) {
    task[id] = instantiate(tmpl[id % perStep], id / perStep, SC, NB);
    for (int sc : {task[id].s0, task[id].s1}) if (sc >= 0) producers.at(sc).push_back(id);
  }
  ReadyGraph g;
  g.indegree.assign(n, 0);
  std::vector<std::vector<int>> succ(n);
  std::vector<int> pred;
  for (int id = 0; id < n; id++) {
    pred.clear();
    const int wc[2] = {task[id].w0c, task[id].w1c}, wv[2] = {task[id].w0v, task[id].w1v};
    for (int k = 0; k < 2; k++) {
      if (wc[k] < 0 || wv[k] == 0) continue;
      const auto& p = producers.at(wc[k]);
      if (wv[k] != (int)p.size()) {
        std::ostringstream m; m << "queue lowering: task " << id << " waits counter " << wc[k] << " >= " << wv[k] << " but it has " << p.size() << " producers (partial threshold)";
        throw std::runtime_error(m.str());
      }
      pred.insert(pred.end(), p.begin(), p.end());
    }
    std::sort(pred.begin(), pred.end()); pred.erase(std::unique(pred.begin(), pred.end()), pred.end());
    for (int p : pred) {
      if (p == id) throw std::runtime_error("queue lowering: task depends on itself");
      succ[p].push_back(id); g.indegree[id]++;
    }
  }
  std::vector<int> left = g.indegree, work;
  for (int i = 0; i < n; i++) if (!left[i]) work.push_back(i);
  g.initial = work;
  int visited = 0;
  while (!work.empty()) { int id = work.back(); work.pop_back(); visited++; for (int nx : succ[id]) if (--left[nx] == 0) work.push_back(nx); }
  if (visited != n) { std::ostringstream m; m << "queue lowering: dependency cycle, " << n - visited << " of " << n << " tasks unreachable"; throw std::runtime_error(m.str()); }
  g.offsets.push_back(0);
  for (auto& row : succ) { g.successors.insert(g.successors.end(), row.begin(), row.end()); g.offsets.push_back((int)g.successors.size()); }
  return g;
}

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- self-test: every transpose combo of the tile GEMM vs CPU fp32 (TF32 tolerance)
static int selftest() {
  CK(cudaFuncSetAttribute(phase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  const int M = 128, N = 192;
  Rng r{777};
  int bad = 0;
  for (int K : {KC, 2 * KC, 3 * KC, 256}) for (int combo = 0; combo < 4; combo++) {
    bool ta = combo & 1, tb = combo & 2;
    std::vector<float> a(M * K), b(K * N), c(M * N), ref(M * N, 0.f);
    for (auto& v : a) v = r.next(); for (auto& v : b) v = r.next();
    // logical A is MxK, B is KxN; physical layouts depend on ta/tb
    auto Aat = [&](int m, int k) -> float& { return ta ? a[k * M + m] : a[m * K + k]; };
    auto Bat = [&](int k, int n) -> float& { return tb ? b[n * K + k] : b[k * N + n]; };
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) { float x = Aat(m, k); for (int n = 0; n < N; n++) ref[m * N + n] += x * Bat(k, n); }
    float *da, *db, *dc; CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, c.size() * 4));
    CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
    Buf hb[3] = {{da, ta ? M : K}, {db, tb ? K : N}, {dc, N}};
    CK(cudaMemcpyToSymbol(g_buf, hb, sizeof hb));
    std::vector<Instr> t;
    for (int m = 0; m < M / TILE; m++) for (int n = 0; n < N / TILE; n++) {
      Instr i; memset(&i, 0, sizeof i); i.op = OP_GEMM; i.flags = (ta ? F_TA : 0) | (tb ? F_TB : 0); i.a = 0; i.b = 1; i.c = 2; i.aux = -1;
      i.w0c = i.w1c = i.s0 = i.s1 = -1; i.m0 = m * TILE; i.n0 = n * TILE; i.K = K; t.push_back(i);
    }
    Instr* dt; CK(cudaMalloc(&dt, t.size() * sizeof(Instr))); CK(cudaMemcpy(dt, t.data(), t.size() * sizeof(Instr), cudaMemcpyHostToDevice));
    { KP kp{0, 1, (int)t.size()}; CK(cudaMemcpyToSymbol(g_kp, &kp, sizeof kp)); }
    phase_kernel<<<(unsigned)t.size(), NT, DSMEM>>>(dt, 0, 0, nullptr); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost));
    double me = 0; for (int i = 0; i < M * N; i++) me = std::max(me, (double)fabs(c[i] - ref[i]));
    printf("selftest K=%d TA=%d TB=%d  max abs err %.3e [c0 %.3f ref0 %.3f]  %s\n", K, ta, tb, me, c[0], ref[0], me < 5e-2 ? "ok" : "BAD");
    if (me >= 5e-2) { bad++;
      if (K == KC && combo == 0) for (int bm = 0; bm < M / 16; bm++) { for (int bn = 0; bn < N / 16; bn++) { double e = 0; for (int i = 0; i < 16; i++) for (int j = 0; j < 16; j++) e = std::max(e, (double)fabs(c[(bm * 16 + i) * N + bn * 16 + j] - ref[(bm * 16 + i) * N + bn * 16 + j])); printf("%c", e < 5e-2 ? '.' : 'X'); } printf("\n"); } }
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dt);
  }
  return bad;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--selftest") return selftest() ? 2 : 0;
  int steps = 1000, B = 128, D = 128, Hd = 256, O = 64, NB = 4, ref_steps = 100, R = 1, reps = 3;
  double watchdog_ms = 5000;
  bool inject_fault = false;   // watchdog test: with --modes claim|queue, make one task unsatisfiable; the run must end in FAIL (watchdog)
  std::string modes_arg = "seq,seqgraph,batched,claim,queue";
  float lr = 0.05f;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto nx = [&]() { return atof(argv[++i]); };
    if (a == "--steps") steps = nx(); else if (a == "--B") B = nx(); else if (a == "--D") D = nx();
    else if (a == "--H") Hd = nx(); else if (a == "--O") O = nx(); else if (a == "--nb") NB = nx();
    else if (a == "--runs") R = nx(); else if (a == "--lr") lr = nx(); else if (a == "--ref-steps") ref_steps = nx();
    else if (a == "--reps") reps = nx(); else if (a == "--watchdog-ms") watchdog_ms = nx();
    else if (a == "--modes") modes_arg = argv[++i]; else if (a == "--quiet") g_verbose = false;
    else if (a == "--inject-fault") inject_fault = true;
    else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
  }
  if (reps < 1 || steps < 1 || R < 1) { fprintf(stderr, "--reps, --steps and --runs must be >= 1\n"); return 1; }
  if (B % TILE || D % TILE || Hd % TILE || O % TILE || B % KC || D % KC || Hd % KC || O % KC) { fprintf(stderr, "dims must be multiples of max(TILE=%d, KC=%d)\n", TILE, KC); return 1; }
  int nRB = B / TILE, nHC = Hd / TILE, nOC = O / TILE, nDC = D / TILE;
  // run r trains the same model from the same init with its own learning rate (an LR sweep)
  std::vector<float> lrs(R); for (int r = 0; r < R; r++) lrs[r] = lr * (R == 1 ? 1.f : 0.25f + 1.5f * r / (R - 1));

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

  // ---- CPU reference (fp32) for a given lr
  int rs = std::min(steps, ref_steps);
  auto cpu_ref = [&](float rlr) {
    std::vector<float> L, rW1 = W1, rW2 = W2;
    std::vector<float> h(B * Hd), dY(B * O), dH(B * Hd), g1(D * Hd), g2(Hd * O);
    for (int s = 0; s < rs; s++) {
      int k = s % NB;
      std::fill(h.begin(), h.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) h[i * Hd + j] += x * rW1[kk * Hd + j]; }
      for (auto& v : h) v = v > 0 ? v : 0;
      std::fill(dY.begin(), dY.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) dY[i * O + j] += x * rW2[kk * O + j]; }
      double Ls = 0;
      for (int i = 0; i < B * O; i++) { float d = dY[i] - T[k][i]; Ls += (double)d * d; dY[i] = d * 2.f / (B * O); }
      L.push_back((float)(Ls / (B * O)));
      std::fill(g2.begin(), g2.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float x = h[i * Hd + kk]; for (int j = 0; j < O; j++) g2[kk * O + j] += x * dY[i * O + j]; }
      for (int i = 0; i < B; i++) for (int kk = 0; kk < Hd; kk++) { float a = 0; for (int j = 0; j < O; j++) a += dY[i * O + j] * rW2[kk * O + j]; dH[i * Hd + kk] = h[i * Hd + kk] > 0 ? a : 0; }
      std::fill(g1.begin(), g1.end(), 0.f);
      for (int i = 0; i < B; i++) for (int kk = 0; kk < D; kk++) { float x = X[k][i * D + kk]; for (int j = 0; j < Hd; j++) g1[kk * Hd + j] += x * dH[i * Hd + j]; }
      for (int i = 0; i < D * Hd; i++) rW1[i] -= rlr * g1[i];
      for (int i = 0; i < Hd * O; i++) rW2[i] -= rlr * g2[i];
    }
    return L;
  };
  std::vector<float> ref0 = cpu_ref(lrs[0]), refN = R > 1 ? cpu_ref(lrs[R - 1]) : ref0;

  // ---- device buffers: shared data X_k, T_k; per-run {W1, W2, H1, dY, dH1}
  auto idX = [&](int k) { return k; };
  auto idT = [&](int k) { return NB + k; };
  auto idW1 = [&](int r) { return 2 * NB + 5 * r; };
  auto idW2 = [&](int r) { return 2 * NB + 5 * r + 1; };
  auto idH1 = [&](int r) { return 2 * NB + 5 * r + 2; };
  auto idDY = [&](int r) { return 2 * NB + 5 * r + 3; };
  auto idDH = [&](int r) { return 2 * NB + 5 * r + 4; };
  int nbuf = 2 * NB + 5 * R;
  if (nbuf > 256) { fprintf(stderr, "too many buffers\n"); return 1; }
  std::vector<Buf> hb(nbuf);
  auto dmalloc = [&](size_t n) { float* p; CK(cudaMalloc(&p, n * 4)); CK(cudaMemset(p, 0, n * 4)); return p; };
  for (int k = 0; k < NB; k++) {
    hb[idX(k)] = {dmalloc(B * D), D}; CK(cudaMemcpy(hb[idX(k)].p, X[k].data(), B * D * 4, cudaMemcpyHostToDevice));
    hb[idT(k)] = {dmalloc(B * O), O}; CK(cudaMemcpy(hb[idT(k)].p, T[k].data(), B * O * 4, cudaMemcpyHostToDevice));
  }
  for (int r = 0; r < R; r++) {
    hb[idW1(r)] = {dmalloc(D * Hd), Hd}; hb[idW2(r)] = {dmalloc(Hd * O), O};
    hb[idH1(r)] = {dmalloc(B * Hd), Hd}; hb[idDY(r)] = {dmalloc(B * O), O}; hb[idDH(r)] = {dmalloc(B * Hd), Hd};
  }
  CK(cudaMemcpyToSymbol(g_buf, hb.data(), nbuf * sizeof(Buf)));

  // ---- tape builder. Per step and run: H1 -> Y -> DH1 -> DW2(+SGD) -> DW1(+SGD); counters give fine-grained deps.
  constexpr int NPH = 5;
  enum { P_H1, P_Y, P_DH1, P_DW2, P_DW1 };
  int CPS = NPH * (nRB + 1);
  int SC = R * CPS;   // counters per step
  auto ctr = [&](int r, int s, int ph, int rb) { return (s * R + r) * CPS + ph * (nRB + 1) + (rb < 0 ? nRB : rb); };
  auto mk = [&]() { Instr i; memset(&i, 0, sizeof i); i.aux = -1; i.w0c = i.w1c = i.s0 = i.s1 = -1; return i; };
  // Template for step 0 (waits on the previous step are flagged and dropped at s==0 on the device).
  auto emit_phase = [&](std::vector<Instr>& tape, int r, int ph) {
    const int s = 0; float rlr = lrs[r];
    if (ph == P_H1) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_RELU; i.a = idX(0); i.b = idW1(r); i.c = idH1(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = D; i.xflags = 4 | 1 | 2;
      i.w0c = ctr(r, s, P_DW1, -1); i.w0v = nDC * nHC; i.w1c = ctr(r, s, P_DW2, -1); i.w1v = nHC * nOC;
      i.s0 = ctr(r, s, P_H1, rb); i.s1 = ctr(r, s, P_H1, -1); tape.push_back(i);
    }
    if (ph == P_Y) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nOC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_MSE; i.a = idH1(r); i.b = idW2(r); i.c = idDY(r); i.aux = idT(0); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = Hd; i.xflags = 8 | 2;
      i.scale = 2.f / (B * O); i.loss_idx = r * steps;
      i.w0c = ctr(r, s, P_H1, rb); i.w0v = nHC; i.w1c = ctr(r, s, P_DW2, -1); i.w1v = nHC * nOC;
      i.s0 = ctr(r, s, P_Y, rb); i.s1 = ctr(r, s, P_Y, -1); tape.push_back(i);
    }
    if (ph == P_DH1) for (int rb = 0; rb < nRB; rb++) for (int cn = 0; cn < nHC; cn++) {
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TB | F_RELUMASK; i.a = idDY(r); i.b = idW2(r); i.c = idDH(r); i.aux = idH1(r); i.m0 = rb * TILE; i.n0 = cn * TILE; i.K = O;
      i.w0c = ctr(r, s, P_Y, rb); i.w0v = nOC; i.s1 = ctr(r, s, P_DH1, -1); tape.push_back(i);
    }
    if (ph == P_DW2) for (int m = 0; m < nHC; m++) for (int n = 0; n < nOC; n++) {   // W2 -= lr H1^T dY (after all dH1 tiles read W2)
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idH1(r); i.b = idDY(r); i.c = idW2(r); i.m0 = m * TILE; i.n0 = n * TILE; i.K = B; i.scale = rlr;
      i.w0c = ctr(r, s, P_Y, -1); i.w0v = nRB * nOC; i.w1c = ctr(r, s, P_DH1, -1); i.w1v = nRB * nHC;
      i.s1 = ctr(r, s, P_DW2, -1); tape.push_back(i);
    }
    if (ph == P_DW1) for (int m = 0; m < nDC; m++) for (int n = 0; n < nHC; n++) {   // W1 -= lr X^T dH1
      Instr i = mk(); i.op = OP_GEMM; i.flags = F_TA | F_SGD; i.a = idX(0); i.b = idDH(r); i.c = idW1(r); i.m0 = m * TILE; i.n0 = n * TILE; i.K = B; i.scale = rlr; i.xflags = 4;
      i.w0c = ctr(r, s, P_DH1, -1); i.w0v = nRB * nHC; i.s1 = ctr(r, s, P_DW1, -1); tape.push_back(i);
    }
  };
  // one-step template, phase-major then run: [ph][r] segments. Batched baseline launches a whole phase (all runs);
  // sequential baseline launches one (run, phase) segment.
  std::vector<Instr> tapeP; std::vector<int> offP(NPH), nP(NPH), nPR(NPH);
  std::vector<std::vector<int>> offPR(NPH, std::vector<int>(R));
  for (int ph = 0; ph < NPH; ph++) {
    offP[ph] = (int)tapeP.size();
    for (int r = 0; r < R; r++) { offPR[ph][r] = (int)tapeP.size(); emit_phase(tapeP, r, ph); }
    nP[ph] = (int)tapeP.size() - offP[ph]; nPR[ph] = nP[ph] / R;
  }
  int perStep = (int)tapeP.size();
  int ntape = perStep * steps, nctr = steps * SC;
  printf("model B=%d D=%d H=%d O=%d  runs=%d steps=%d  tile=%d kc=%d stg=%d  tape=%d instrs executed, %d-entry step template (%.2f MB)\n", B, D, Hd, O, R, steps, TILE, KC, STG, ntape, perStep, perStep * sizeof(Instr) / 1e6);

  // ---- modes: the four conventional/claim executions of the tape plus the ready queue. --modes picks a subset.
  enum { M_SEQ, M_SEQGRAPH, M_BATCHED, M_CLAIM, M_QUEUE, NMODE };
  const char* key[NMODE] = {"seq", "seqgraph", "batched", "claim", "queue"};
  const char* nm[NMODE] = {"sequential, launch-per-op", "sequential, CUDA graph/run", "batched across runs (vmap-like graph)", "KVM: claim-and-wait", "KVM: ready queue"};
  bool on[NMODE] = {};
  { std::stringstream ss(modes_arg); std::string k;
    while (std::getline(ss, k, ',')) { int m = 0; while (m < NMODE && k != key[m]) m++;
      if (m == NMODE) { fprintf(stderr, "unknown mode '%s' in --modes (seq,seqgraph,batched,claim,queue)\n", k.c_str()); return 1; } on[m] = true; } }
  if (std::count(on, on + NMODE, true) < 1) { fprintf(stderr, "--modes selects nothing\n"); return 1; }

  if (inject_fault && on[M_CLAIM] == on[M_QUEUE]) { fprintf(stderr, "--inject-fault needs exactly one of the modes claim, queue\n"); return 1; }
  if (inject_fault && on[M_CLAIM]) { tapeP.back().w0v++; LOGI("[fault] last template instruction now waits for one more completion than will ever signal it\n"); }
  Instr* dtapeP; int *dhead, *dctr; float* dloss; Diag* ddiag;
  CK(cudaMalloc(&dtapeP, perStep * sizeof(Instr))); CK(cudaMemcpy(dtapeP, tapeP.data(), perStep * sizeof(Instr), cudaMemcpyHostToDevice));
  { KP kp{SC, NB, perStep}; CK(cudaMemcpyToSymbol(g_kp, &kp, sizeof kp)); }
  CK(cudaMalloc(&dhead, 4)); CK(cudaMalloc(&dctr, (size_t)nctr * 4)); CK(cudaMalloc(&dloss, (size_t)R * steps * 4)); CK(cudaMalloc(&ddiag, sizeof(Diag)));

  // ---- ready-queue state: DAG lowered once on the host, queue arrays reset before every timed launch.
  ReadyGraph dag; QueueDev qd{}; int* qbuf = nullptr;
  if (on[M_QUEUE]) {
    double tl0 = now_ms();
    try { dag = lower_to_dag(tapeP, steps, SC, NB); } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
    size_t ne = dag.successors.size();
    if (ne > (size_t)INT32_MAX) { fprintf(stderr, "queue: %zu edges overflow int offsets\n", ne); return 1; }
    int* offs; int* succ;   // [offsets n+1][successors ne][remaining n][task n][ready n][head tail done]
    size_t words = (size_t)(ntape + 1) + std::max<size_t>(ne, 1) + 3 * (size_t)ntape + 3;
    CK(cudaMalloc(&qbuf, words * 4));
    offs = qbuf; succ = offs + ntape + 1;
    qd = {offs, succ, succ + std::max<size_t>(ne, 1), nullptr, nullptr, nullptr, nullptr, nullptr};
    qd.task = qd.remaining + ntape; qd.ready = qd.task + ntape; qd.head = qd.ready + ntape; qd.tail = qd.head + 1; qd.done = qd.tail + 1;
    CK(cudaMemcpy(offs, dag.offsets.data(), dag.offsets.size() * 4, cudaMemcpyHostToDevice));
    if (ne) CK(cudaMemcpy(succ, dag.successors.data(), ne * 4, cudaMemcpyHostToDevice));
    LOGI("[queue] lowered %d tasks, %zu edges (%.1f/task), %zu initially ready, %.1f MB device, %.0f ms host\n", ntape, ne, (double)ne / ntape, dag.initial.size(), words * 4 / 1e6, now_ms() - tl0);
  }

  CK(cudaFuncSetAttribute(vm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  CK(cudaFuncSetAttribute(queue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  CK(cudaFuncSetAttribute(phase_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DSMEM));
  int dev; cudaGetDevice(&dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  int occ, qocc; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, vm_kernel, NT, DSMEM)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&qocc, queue_kernel, NT, DSMEM));
  int grid = prop.multiProcessorCount * occ, qgrid = prop.multiProcessorCount * qocc;
  int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
  printf("%s  SMs=%d  driver CUDA %d.%d  runtime %d.%d  vm grid=%d (occ %d)  queue grid=%d (occ %d)  watchdog %.0f ms\n", prop.name, prop.multiProcessorCount, drv / 1000, drv % 1000 / 10, rt / 1000, rt % 1000 / 10, grid, occ, qgrid, qocc, watchdog_ms);

  auto reset = [&]() {
    for (int r = 0; r < R; r++) {
      CK(cudaMemcpy(hb[idW1(r)].p, W1.data(), W1.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(hb[idW2(r)].p, W2.data(), W2.size() * 4, cudaMemcpyHostToDevice));
    }
    CK(cudaMemset(dhead, 0, 4)); CK(cudaMemset(dctr, 0, (size_t)nctr * 4)); CK(cudaMemset(dloss, 0, (size_t)R * steps * 4));
    Diag d0{}; d0.limit_ns = (unsigned long long)(watchdog_ms * 1e6); CK(cudaMemcpy(ddiag, &d0, sizeof d0, cudaMemcpyHostToDevice));
    if (on[M_QUEUE]) {
      std::vector<int> rem = dag.indegree;
      if (inject_fault) { rem[ntape - 1]++; }   // one predecessor that does not exist: the last task is never released
      CK(cudaMemcpy(qd.remaining, rem.data(), (size_t)ntape * 4, cudaMemcpyHostToDevice));
      CK(cudaMemset(qd.ready, 0, (size_t)ntape * 4));
      CK(cudaMemcpy(qd.task, dag.initial.data(), dag.initial.size() * 4, cudaMemcpyHostToDevice));
      std::vector<int> ones(dag.initial.size(), 1); CK(cudaMemcpy(qd.ready, ones.data(), ones.size() * 4, cudaMemcpyHostToDevice));
      int cur[3] = {0, (int)dag.initial.size(), 0}; CK(cudaMemcpy(qd.head, cur, sizeof cur, cudaMemcpyHostToDevice));   // head, tail, done are adjacent
    }
    CK(cudaDeviceSynchronize());
  };
  auto launch_seq = [&](cudaStream_t st, int rlo, int rhi) {   // one model at a time: run r, step s, phase ph
    for (int r = rlo; r < rhi; r++) for (int s = 0; s < steps; s++) for (int ph = 0; ph < NPH; ph++) phase_kernel<<<nPR[ph], NT, DSMEM, st>>>(dtapeP, offPR[ph][r], s, dloss);
  };
  auto launch_batched = [&](cudaStream_t st) { for (int s = 0; s < steps; s++) for (int ph = 0; ph < NPH; ph++) phase_kernel<<<nP[ph], NT, DSMEM, st>>>(dtapeP, offP[ph], s, dloss); };
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  // Runs one mode from a reset state; if the watchdog fired, reports why and fails the run.
  bool watchdog_fired = false;
  auto timed = [&](int mode, auto fn) {
    reset(); cudaEventRecord(e0); fn(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    Diag d; CK(cudaMemcpy(&d, ddiag, sizeof d, cudaMemcpyDeviceToHost));
    if (d.tripped) {
      watchdog_fired = true;
      fprintf(stderr, "[watchdog] mode '%s' gave up after %.0f ms in block %d\n", key[mode], watchdog_ms, d.block);
      if (mode == M_CLAIM) fprintf(stderr, "[watchdog]   claim VM: instruction %d of %d waits counter %d >= %d, saw %d\n", d.idx, d.n, d.ctr, d.want, d.seen);
      else fprintf(stderr, "[watchdog]   ready queue: ticket %d never published; head %d tail %d done %d of %d tasks\n", d.idx, d.head, d.tail, d.done, d.n);
    }
    return (double)ms;
  };
  struct Out { std::vector<float> L, w; };
  auto fetch = [&]() {
    Out o; o.L.resize((size_t)R * steps);
    CK(cudaMemcpy(o.L.data(), dloss, o.L.size() * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < R; r++) { std::vector<float> a(W1.size()), b(W2.size());
      CK(cudaMemcpy(a.data(), hb[idW1(r)].p, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), hb[idW2(r)].p, b.size() * 4, cudaMemcpyDeviceToHost));
      o.w.insert(o.w.end(), a.begin(), a.end()); o.w.insert(o.w.end(), b.begin(), b.end()); }
    return o;
  };

  // graphs: (a) each run's whole training, replayed per run  (b) all runs batched per phase
  cudaGraphExec_t xB; std::vector<cudaGraphExec_t> xRun(R);
  { cudaStream_t cs; cudaStreamCreate(&cs);
    if (on[M_BATCHED]) { cudaGraph_t g; cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_batched(cs); CK(cudaStreamEndCapture(cs, &g)); CK(cudaGraphInstantiate(&xB, g, 0)); }
    if (on[M_SEQGRAPH]) for (int r = 0; r < R; r++) { cudaGraph_t g; cudaStreamBeginCapture(cs, cudaStreamCaptureModeGlobal); launch_seq(cs, r, r + 1); CK(cudaStreamEndCapture(cs, &g)); CK(cudaGraphInstantiate(&xRun[r], g, 0)); } }

  std::function<void()> fn[NMODE] = {
    [&]() { launch_seq(0, 0, R); },
    [&]() { for (int r = 0; r < R; r++) CK(cudaGraphLaunch(xRun[r], 0)); },
    [&]() { CK(cudaGraphLaunch(xB, 0)); },
    [&]() { vm_kernel<<<grid, NT, DSMEM>>>(dtapeP, ntape, dhead, dctr, dloss, nullptr, ddiag); },
    [&]() { queue_kernel<<<qgrid, NT, DSMEM>>>(dtapeP, ntape, qd, dloss, ddiag); }};

  // One warm-up pass, then `reps` timed passes with the modes interleaved, so a ratio compares neighbours in time and the
  // spread across passes shows how much the shared GPU moved. Weights of every pass are checked bitwise against the first
  // enabled mode's first pass: the same tape must give the same bits however it is scheduled, so any difference is a race.
  int ref_mode = 0; while (!on[ref_mode]) ref_mode++;
  std::vector<double> tm[NMODE]; Out o[NMODE]; long long mismatches[NMODE] = {}; double e_mode = 0;
  Out ref_w;
  for (int pass = -1; pass < reps; pass++) for (int m = 0; m < NMODE; m++) if (on[m]) {
    double ms = timed(m, fn[m]);
    if (watchdog_fired) { printf("FAIL (watchdog)\n"); return 2; }
    Out cur = fetch();
    if (pass < 0 && m == ref_mode) ref_w = cur;
    long long bad = 0; for (size_t j = 0; j < cur.w.size(); j++) if (memcmp(&cur.w[j], &ref_w.w[j], 4)) { bad++; e_mode = std::max(e_mode, (double)fabs(cur.w[j] - ref_w.w[j])); }
    mismatches[m] += bad;
    LOGI("[pass %2d] %-40s %9.3f ms  weight words differing from '%s': %lld\n", pass, nm[m], ms, key[ref_mode], bad);
    if (pass >= 0) tm[m].push_back(ms);
    o[m] = std::move(cur);
  }
  auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  auto bestt = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };

  double tot = (double)R * steps;
  printf("\n%-40s %10s %10s %10s %14s %11s\n", "mode", "min ms", "median ms", "max ms", "us/step/run", "run-steps/s");
  for (int m = 0; m < NMODE; m++) if (on[m]) printf("%-40s %10.3f %10.3f %10.3f %14.2f %11.0f\n", nm[m], bestt(tm[m]), med(tm[m]), *std::max_element(tm[m].begin(), tm[m].end()), bestt(tm[m]) * 1000 / tot, tot / (bestt(tm[m]) / 1000));
  // ratio of time(baseline)/time(mode) per pass: >1 means the mode is faster. Printed as median [min, max] over passes.
  auto ratio = [&](int base, int m) {
    std::vector<double> r; for (int i = 0; i < reps; i++) r.push_back(tm[base][i] / tm[m][i]);
    printf("%s vs %s: %.2fx  [min %.2fx, max %.2fx] over %d passes\n", key[m], key[base], med(r), *std::min_element(r.begin(), r.end()), *std::max_element(r.begin(), r.end()), reps); };
  if (on[M_CLAIM] && on[M_BATCHED]) ratio(M_BATCHED, M_CLAIM);
  if (on[M_QUEUE] && on[M_BATCHED]) ratio(M_BATCHED, M_QUEUE);
  if (on[M_QUEUE] && on[M_CLAIM]) ratio(M_CLAIM, M_QUEUE);

  // ---- correctness
  auto relerr = [&](const float* a, const std::vector<float>& ref) { double m = 0; for (int i = 0; i < rs; i++) m = std::max(m, (double)fabs(a[i] - ref[i]) / (fabs(ref[i]) + 1e-6)); return m; };
  double e_ref = 0; bool dec = true; long long total_bad = 0;
  for (int m = 0; m < NMODE; m++) if (on[m]) {
    e_ref = std::max(e_ref, relerr(&o[m].L[0], ref0)); e_ref = std::max(e_ref, relerr(&o[m].L[(size_t)(R - 1) * steps], refN));
    for (int r = 0; r < R; r++) dec = dec && o[m].L[(size_t)r * steps + steps - 1] < o[m].L[(size_t)r * steps];
    total_bad += mismatches[m];
  }
  int kv = on[M_CLAIM] ? M_CLAIM : ref_mode;
  printf("\nloss run0: %.5f -> %.5f (lr %.4f)", o[kv].L[0], o[kv].L[steps - 1], lrs[0]);
  if (R > 1) printf("   run%d: %.5f -> %.5f (lr %.4f)", R - 1, o[kv].L[(size_t)(R - 1) * steps], o[kv].L[(size_t)(R - 1) * steps + steps - 1], lrs[R - 1]);
  printf("\nmax rel loss err vs cpu fp32 (runs 0 and %d, first %d steps, all modes): %.2e\n", R - 1, rs, e_ref);
  printf("final weights, %d modes x %d passes vs '%s': %lld differing words, max |dW| %.2e (must be exactly 0)\n", (int)std::count(on, on + NMODE, true), reps + 1, key[ref_mode], total_bad, e_mode);
  bool ok = e_ref < 2e-2 && total_bad == 0 && dec;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 2;
}
