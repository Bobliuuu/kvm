// Adversarial litmus harness for the KVM executor's lock-free protocols. It drives the REAL device functions from
// include/kvm_exec.cuh (kvm_release_task / kvm_park / kvm_publish / kvm_enqueue / kvm_pop) in isolation, with thousands of
// CTAs, millions of rounds and randomized per-round delays. build.sh also builds MUTANTS (header copies with one ordering
// step removed): a litmus that cannot fail proves nothing, so every property has at least one mutant that must be caught.
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude tests/litmus/litmus.cu -o litmus
//   ./litmus <test> [rounds_k] [launches] [seed]       tests: sb dekker1 dekker3 mp queue all   (queue: pops = rounds_k*20k)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#ifdef LIT_NOFENCE            // ablation of every __threadfence in the production header (macro applies after cuda_runtime.h)
#define __threadfence() ((void)0)
#endif
#include "kvm_exec.cuh"
#define NSLOTS_SET(nt) (S.nslots = 0)   /* slot-recycling executor: no free ring here (kvm_init_kernel must not touch fcells) */
__device__ __forceinline__ int lit_pop(const KvmDevState& S) { int how; return kvm_pop(S, kvm_now(), -1, how); }   // scheduler off: one class

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(3); } } while (0)

__device__ __forceinline__ unsigned lit_hash(unsigned a, unsigned b, unsigned c) {
  unsigned x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 1u) * 0xC2B2AE35u;
  x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15; return x;
}
__device__ __forceinline__ void lit_delay(unsigned cycles) { long long t = clock64(); while (clock64() - t < (long long)cycles) {} }
// two-sided (or n-sided) rendezvous per round, then a randomized skew so both orders and tight overlaps are covered
__device__ __forceinline__ bool lit_rendezvous(int* arr, int r, int n) {
  atomicAdd(arr + r, 1);
  for (long long i = 0; *(volatile int*)(arr + r) < n; ++i) if (i > (1ll << 24)) return false;
  return true;
}
__device__ __forceinline__ unsigned lit_skew(unsigned seed, int r, int role, int nroles) {
  unsigned h = lit_hash(seed, r, role);
  switch ((h >> 28) & 7) {          // delay profile of this round
    case 0: return 0;
    case 1: return (role == 0 ? 600 : 0) + (h & 63);      // role 0 late
    case 2: return (role != 0 ? 600 : 0) + (h & 63);      // others late
    case 3: return h & 1023;
    case 4: return (h & 127);
    case 5: return (role & 1 ? 1500 : 0) + (h & 255);
    case 6: return (role & 1 ? 0 : 1500) + (h & 255);
    default: return (h >> 4) & 4095;
  }
}
__device__ __forceinline__ void* gp(const void* p) { return (void*)p; }

// ================= SB: store-buffering, the shape of the Dekker argument ================= (v: see kernel)
// v0 st.relaxed/ld.relaxed no fence | v1 fence.sc between | v2 fence.acq_rel between | v3 atom.exch.acq_rel + ld.acquire (no fence)
// v4 atom.exch.acq_rel + fence.sc + ld.acquire  (the production sequence) | v5 st.relaxed + fence.sc.gpu + ld.relaxed on same line pairs
__global__ void k_sb(int v, int R, unsigned seed, int* X, int* Y, int* R0, int* R1, int* arr, int* weak_abort) {
  int pair = blockIdx.x >> 1, role = blockIdx.x & 1, npairs = gridDim.x >> 1;
  if (threadIdx.x) return;
  for (int r = pair; r < R; r += npairs) {
    if (!lit_rendezvous(arr, r, 2)) { atomicAdd(weak_abort, 1); return; }
    lit_delay(lit_skew(seed, r, role, 2));
    int* mine = role ? Y : X; int* other = role ? X : Y; int got;
    switch (v) {
      case 0: asm volatile("st.relaxed.gpu.global.b32 [%0],1;" ::"l"(mine + r) : "memory"); asm volatile("ld.relaxed.gpu.global.b32 %0,[%1];" : "=r"(got) : "l"(other + r) : "memory"); break;
      case 1: asm volatile("st.relaxed.gpu.global.b32 [%0],1;\n fence.sc.gpu;" ::"l"(mine + r) : "memory"); asm volatile("ld.relaxed.gpu.global.b32 %0,[%1];" : "=r"(got) : "l"(other + r) : "memory"); break;
      case 2: asm volatile("st.relaxed.gpu.global.b32 [%0],1;\n fence.acq_rel.gpu;" ::"l"(mine + r) : "memory"); asm volatile("ld.relaxed.gpu.global.b32 %0,[%1];" : "=r"(got) : "l"(other + r) : "memory"); break;
      case 3: atom_exch(mine + r, 1); got = ld_acq(other + r); break;
      case 4: atom_exch(mine + r, 1); __threadfence(); got = ld_acq(other + r); break;
      default: asm volatile("st.weak.global.b32 [%1],1;\n ld.weak.global.b32 %0,[%2];" : "=r"(got) : "l"(mine + r), "l"(other + r) : "memory"); break;
    }
    (role ? R1 : R0)[r] = got;
  }
}

// ================= Dekker: real kvm_release_task (waiter) vs real kvm_publish (publisher) =================
// nw=1: waiter A needs v>=1, publisher P1 sets 1.  nw=3: A needs 1; B,C need 2; P1 sets 1; P2 (waits for 1) sets 2 (re-park path).
// Roles per round: CTA 2g = waiters (warp i = waiter i), CTA 2g+1 = publishers (warp 0 = P1, warp 1 = P2).
__global__ void k_dekker(KvmDevState S, int nw, int R, unsigned seed, int* arr, int* abort_flag) {
  int pair = blockIdx.x >> 1, role = blockIdx.x & 1, npairs = gridDim.x >> 1;
  int warp = threadIdx.x >> 5; if (threadIdx.x & 31) return;
  int nparts = nw + (nw == 1 ? 1 : 2);
  const bool part = (role == 0 && warp < nw) || (role == 1 && (warp == 0 || (warp == 1 && nw == 3)));
  if (!part) return;
  for (int r = pair; r < R; r += npairs) {
    int base = nw * r;
    if (!lit_rendezvous(arr, r, nparts)) { atomicAdd(abort_flag, 1); return; }
    int myrole = role ? 4 + warp : warp;
    if (role == 0 && warp < nw) {
      lit_delay(lit_skew(seed, r, myrole, nparts));
      kvm_release_task(S, base + warp);              // production path: enqueue if satisfied else park (+ Dekker recheck)
    } else if (role == 1 && warp == 0) {
      lit_delay(lit_skew(seed, r, myrole, nparts));
      kvm_publish(S, 0, r, 1);
    } else if (role == 1 && warp == 1 && nw == 3) {
      while (ld_acq(S.version + r) < 1) {}           // graph order: P2 after P1
      lit_delay(lit_skew(seed, r, myrole, nparts));
      kvm_publish(S, 0, r, 2);
    }
  }
}
__global__ void k_drain(KvmDevState S, int* seen, int nslots, int* popped) {
  if (blockIdx.x || threadIdx.x >= 32) return;
  for (;;) {
    int s = lit_pop(S);
    if (s == -1) { s = lit_pop(S); if (s == -1) break; }
    if (threadIdx.x == 0 && s >= 0 && s < nslots) { seen[s]++; (*popped)++; }
  }
}
__global__ void k_init_tasks(KvmDevState S, int nw, int R) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nw * R; i += gridDim.x * blockDim.x) {
    int r = i / nw, w = i % nw;
    KvmTaskDesc t{}; t.wait_var = r; t.wait_ver = (nw == 1 || w == 0) ? 1 : 2; t.priority = 0; t.pub_var = -1;
    S.tasks[i] = t; S.state[i] = 0; S.park_next[i] = -1;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < R; i += gridDim.x * blockDim.x) { S.version[i] = 0; S.park_head[i] = -1; }
}

struct Dev {  // minimal executor state for the lock-free primitives
  KvmDevState S{}; int live_cap = 0, nvars = 0, ntasks = 0;
  void alloc(int cap, int nv, int nt) {
    live_cap = cap; nvars = nv; ntasks = nt; S.live_cap = cap; S.rcap = 4 * cap; S.sp.nc = 1; S.sp.on = 0; S.num_vars = nv; NSLOTS_SET(nt); S.aging_ns = 1ll << 50; S.timeout_ns = ~0ull;
    CK(cudaMalloc(&S.tasks, (size_t)nt * sizeof(KvmTaskDesc))); CK(cudaMalloc(&S.state, nt * 4)); CK(cudaMalloc(&S.park_next, nt * 4));
    CK(cudaMalloc(&S.version, nv * 4)); CK(cudaMalloc(&S.park_head, nv * 4));
    CK(cudaMalloc(&S.cells, (size_t)KVM_NBUCKETS * 4 * cap * sizeof(KvmCell))); CK(cudaMalloc(&S.qctl, KVM_NBUCKETS * sizeof(KvmQCtl)));
    CK(cudaMalloc(&S.trace, (size_t)nt * sizeof(KvmTrace))); CK(cudaMalloc(&S.cnt, sizeof(KvmCounters))); CK(cudaMalloc(&S.herr, sizeof(KvmHostErr)));
  }
  void reset() {
    CK(cudaMemset(S.qctl, 0, KVM_NBUCKETS * sizeof(KvmQCtl))); CK(cudaMemset(S.cnt, 0, sizeof(KvmCounters))); CK(cudaMemset(S.herr, 0, sizeof(KvmHostErr)));
    CK(cudaMemset(S.trace, 0, (size_t)ntasks * sizeof(KvmTrace)));
    kvm_init_kernel<<<256, KVM_THREADS>>>(S); CK(cudaGetLastError());   // production init: cells seq/slot, park_head
    CK(cudaDeviceSynchronize());
  }
  int err() { KvmCounters c; CK(cudaMemcpy(&c, S.cnt, sizeof c, cudaMemcpyDeviceToHost)); return c.err_code; }
};
static int grid_for(void* kern, int threads) {
  int dev, sms, occ; CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, 0)); int g = sms * occ; return g & ~1;
}

static long long g_total_rounds = 0, g_total_bad = 0;
// control=true: a deliberately broken variant that MUST show violations (proves the harness can fail); it is not counted as a bug
static void report(const char* name, long long rounds, long long bad, const char* what, bool control = false) {
  printf("LITMUS %-34s rounds=%lld  %s=%lld  %s\n", name, rounds, what, bad,
         control ? (bad ? "CONTROL: failure detected (harness is sensitive)" : "CONTROL: NOT DETECTED (harness too weak!)") : (bad ? "VIOLATIONS" : "none")); fflush(stdout);
  g_total_rounds += rounds; g_total_bad += control ? (bad ? 0 : 1) : bad;
}

static void run_sb(int v, int Rk, int launches, unsigned seed) {
  int R = Rk * 1000; int* d[6]; for (auto& p : d) CK(cudaMalloc(&p, (size_t)R * 4));
  int* ab; CK(cudaMalloc(&ab, 4));
  int grid = grid_for((void*)k_sb, 32); long long bad = 0, both1 = 0;
  std::vector<int> r0(R), r1(R);
  for (int l = 0; l < launches; ++l) {
    for (int i = 0; i < 5; ++i) CK(cudaMemset(d[i], 0, (size_t)R * 4)); CK(cudaMemset(ab, 0, 4));
    CK(cudaMemset(d[2], 0xff, (size_t)R * 4)); CK(cudaMemset(d[3], 0xff, (size_t)R * 4));
    k_sb<<<grid, 32>>>(v, R, seed + l, d[0], d[1], d[2], d[3], d[4], ab); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(r0.data(), d[2], (size_t)R * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(r1.data(), d[3], (size_t)R * 4, cudaMemcpyDeviceToHost));
    for (int i = 0; i < R; ++i) { if (r0[i] == 0 && r1[i] == 0) ++bad; if (r0[i] == 1 && r1[i] == 1) ++both1; }
  }
  static const char* nm[] = {"relaxed,no-fence", "relaxed+fence.sc", "relaxed+fence.acq_rel", "atom.exch.acq_rel+ld.acq,no-fence", "atom.exch.acq_rel+fence.sc+ld.acq (PROD)", "weak st/ld"};
  char buf[96]; snprintf(buf, sizeof buf, "sb[%s]", nm[v]); report(buf, (long long)R * launches, bad, "both_miss(0,0)", v == 0);
  for (auto p : d) cudaFree(p); cudaFree(ab);
}

static void run_dekker(int nw, int Rk, int launches, unsigned seed) {
  int R = Rk * 1000; int cap = 2; while (cap < R * nw) cap <<= 1;   // bucket 0 must hold every waiter
  Dev D; D.alloc(cap, R, R * nw);
  int *arr, *ab, *seen, *popped; CK(cudaMalloc(&arr, R * 4)); CK(cudaMalloc(&ab, 4)); CK(cudaMalloc(&seen, (size_t)R * nw * 4)); CK(cudaMalloc(&popped, 4));
  int grid = grid_for((void*)k_dekker, 128);
  std::vector<int> h((size_t)R * nw); long long lost = 0, dup = 0, aborted = 0, errs = 0;
  for (int l = 0; l < launches; ++l) {
    D.reset(); k_init_tasks<<<256, 256>>>(D.S, nw, R); CK(cudaGetLastError());
    CK(cudaMemset(arr, 0, R * 4)); CK(cudaMemset(ab, 0, 4)); CK(cudaMemset(seen, 0, (size_t)R * nw * 4)); CK(cudaMemset(popped, 0, 4));
    CK(cudaDeviceSynchronize());
    k_dekker<<<grid, 128>>>(D.S, nw, R, seed + l * 77, arr, ab); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    k_drain<<<1, 32>>>(D.S, seen, R * nw, popped); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), seen, (size_t)R * nw * 4, cudaMemcpyDeviceToHost));
    int a; CK(cudaMemcpy(&a, ab, 4, cudaMemcpyDeviceToHost)); aborted += a; errs += D.err() != 0;
    for (size_t i = 0; i < h.size(); ++i) { if (h[i] == 0) ++lost; else if (h[i] > 1) ++dup; }
  }
  char buf[64]; snprintf(buf, sizeof buf, "dekker%d (real release/park/publish)", nw);
  report(buf, (long long)R * launches, lost + dup + aborted + errs, "lost+dup+abort+err");
  printf("       detail: lost_wakeups=%lld double_enqueue=%lld rendezvous_aborts=%lld device_errors=%lld  waiters=%lld\n", lost, dup, aborted, errs, (long long)R * nw * launches);
}

// ================= MP: publish-before-data. writer CTA stores a block, publishes; reader CTA primes L1 with the OLD data, spins =================
// on the version with the production acquire (thread 0), bar.sync, then ALL threads read with plain loads. variants:
// 0 production kvm_publish   1 mutant: publish via relaxed CAS, no fence (needs -DLIT_MP_WEAK to differ in reader too)
// 2 negative control: thread 0 publishes BEFORE the other threads store (KVM_BREAK_FENCE shape)   3 reader uses relaxed version read
#define MPW 1024
__device__ __forceinline__ unsigned ldg_plain(const unsigned* p) { unsigned v; asm volatile("ld.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__global__ void k_mp(KvmDevState S, int v, int R, unsigned seed, unsigned* data, int* arr, int* bad_words, int* bad_rounds, int* abort_flag) {
  int pair = blockIdx.x >> 1, role = blockIdx.x & 1, npairs = gridDim.x >> 1;
  for (int r = pair; r < R; r += npairs) {
    unsigned* dp = data + (size_t)r * MPW;
    unsigned pat = 0x9E3779B9u * (r + 1);
    if (role == 0) {  // reader (also the slower to arrive in half the rounds)
      unsigned prime = 0;
      for (int i = threadIdx.x; i < MPW; i += blockDim.x) prime += ldg_plain(dp + i);     // old (zero) data now maybe in L1
      __syncthreads();
      if (threadIdx.x == 0 && !lit_rendezvous(arr, r, 2)) { atomicAdd(abort_flag, 1); }
      if (threadIdx.x == 0) {
        lit_delay(lit_skew(seed, r, 0, 2));
        long long it = 0;
        if (v == 3) { while (true) { int x; asm volatile("ld.relaxed.gpu.global.b32 %0,[%1];" : "=r"(x) : "l"(S.version + r) : "memory"); if (x >= 1) break; if (++it > (1ll << 26)) { atomicAdd(abort_flag, 1); break; } } }
        else { while (ld_acq(S.version + r) < 1) if (++it > (1ll << 26)) { atomicAdd(abort_flag, 1); break; } }
      }
      __syncthreads();
      int badw = 0;
      for (int i = threadIdx.x; i < MPW; i += blockDim.x) if (ldg_plain(dp + i) != pat + i) ++badw;
      if (badw) atomicAdd(bad_words, badw);
      if (__syncthreads_or(badw != 0) && threadIdx.x == 0) atomicAdd(bad_rounds, 1);
      __syncthreads();
      if (prime == 0xdeadbeef) dp[0] = 0;  // keep the priming loads alive
    } else {          // writer
      if (threadIdx.x == 0 && !lit_rendezvous(arr, r, 2)) { atomicAdd(abort_flag, 1); }
      __syncthreads();
      lit_delay(lit_skew(seed, r, 1, 2) + (lit_hash(seed, r, threadIdx.x) & 63));
      if (v == 2 && threadIdx.x == 0) kvm_publish(S, 0, r, 1);                  // negative control: version first
      for (int i = threadIdx.x; i < MPW; i += blockDim.x) dp[i] = pat + i;      // plain stores, like the ops
      __syncthreads();
      if (threadIdx.x == 0 && v != 2) {
        if (v == 1) { int o = atomicCAS(S.version + r, 0, 1); (void)o; }        // relaxed CAS, no fence (mutant)
        else kvm_publish(S, 0, r, 1);                                           // production
      }
    }
  }
}
static void run_mp(int v, int Rk, int launches, unsigned seed) {
  int R = Rk * 1000; Dev D; D.alloc(2, R, R);
  unsigned* data; int *arr, *bw, *br, *ab; CK(cudaMalloc(&data, (size_t)R * MPW * 4)); CK(cudaMalloc(&arr, R * 4)); CK(cudaMalloc(&bw, 4)); CK(cudaMalloc(&br, 4)); CK(cudaMalloc(&ab, 4));
  int grid = grid_for((void*)k_mp, 256); long long badw = 0, badr = 0, ab_total = 0;
  for (int l = 0; l < launches; ++l) {
    D.reset(); k_init_tasks<<<256, 256>>>(D.S, 1, R);
    CK(cudaMemset(data, 0, (size_t)R * MPW * 4)); CK(cudaMemset(arr, 0, R * 4)); CK(cudaMemset(bw, 0, 4)); CK(cudaMemset(br, 0, 4)); CK(cudaMemset(ab, 0, 4)); CK(cudaDeviceSynchronize());
    k_mp<<<grid, 256>>>(D.S, v, R, seed + l, data, arr, bw, br, ab); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int a, b, c; CK(cudaMemcpy(&a, bw, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&b, br, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&c, ab, 4, cudaMemcpyDeviceToHost));
    badw += a; badr += b; ab_total += c; (void)D.err();
  }
  static const char* nm[] = {"PROD kvm_publish + acquire", "MUTANT relaxed CAS no fence", "NEGATIVE publish-before-stores", "MUTANT reader relaxed version"};
  char buf[96]; snprintf(buf, sizeof buf, "mp[%s]", nm[v]); report(buf, (long long)R * launches, badr + ab_total, "stale_rounds+aborts", v != 0);
  printf("       detail: stale_words=%lld stale_rounds=%lld aborts=%lld (grid %d CTAs x 256 thr, %d words/round)\n", badw, badr, ab_total, grid, MPW);
}

// ================= Queue: real kvm_enqueue / kvm_pop with token recirculation, wraparound, ABA =================
// ntok = live_cap tokens circulate (the executor's in-flight bound). A popper checks the token was queued exactly once (flag CAS),
// increments a PLAIN per-token counter (lost updates => handoff visibility bug), picks a new priority and re-enqueues.
__global__ void k_queue2(KvmDevState S, int ntok, long long budget, unsigned seed, int* inq, unsigned* payload, long long* pops, int* dbl, int* abort_flag) {
  const int lane = threadIdx.x;
  long long it = 0;
  for (;;) {
    int s = lit_pop(S);                      // warp-wide; every lane gets the same value
    if (s < 0) {
      if (++it > (1ll << 21)) { if (lane == 0) atomicAdd(abort_flag, 1); return; }
      int fin = lane == 0 ? (*(volatile long long*)pops >= budget) : 0;
      if (__shfl_sync(0xffffffffu, fin, 0)) return;       // warp-uniform: kvm_pop is warp-collective
      __nanosleep(lit_hash(seed, blockIdx.x, (unsigned)it) & 63);
      continue;
    }
    it = 0;
    long long n = 0;
    if (lane == 0) {
      n = (long long)atomicAdd((unsigned long long*)pops, 1ull);
      if (atomicCAS(inq + s, 1, 0) != 1) atomicAdd(dbl, 1);
      payload[s] = payload[s] + 1;
      unsigned h = lit_hash(seed, s, (unsigned)n);
      if (h & 1) lit_delay((h >> 3) & 255);
      S.tasks[s].priority = lit_hash(seed + 5, s, (unsigned)n) % KVM_NBUCKETS;
      atomicExch(inq + s, 1);
      kvm_enqueue(S, s);
    }
    __syncwarp();
    if (__shfl_sync(0xffffffffu, (int)(n >= budget), 0)) return;   // budget spent (the token above is back in its queue)
  }
}
__global__ void k_queue_init(KvmDevState S, int ntok, int* inq, unsigned* payload) {
  if (blockIdx.x || threadIdx.x) return;
  for (int i = 0; i < ntok; ++i) { KvmTaskDesc t{}; t.priority = i % KVM_NBUCKETS; S.tasks[i] = t; inq[i] = 1; payload[i] = 0; kvm_enqueue(S, i); }
}
static void run_queue(int cap, int blocks, long long budget_k, unsigned seed) {
  Dev D; D.alloc(cap, 1, cap); D.reset();
  int *inq, *dbl, *ab, *seen, *popped; unsigned* pay; long long* pops;
  CK(cudaMalloc(&inq, cap * 4)); CK(cudaMalloc(&pay, cap * 4)); CK(cudaMalloc(&pops, 8)); CK(cudaMalloc(&dbl, 4)); CK(cudaMalloc(&ab, 4)); CK(cudaMalloc(&seen, cap * 4)); CK(cudaMalloc(&popped, 4));
  CK(cudaMemset(pops, 0, 8)); CK(cudaMemset(dbl, 0, 4)); CK(cudaMemset(ab, 0, 4)); CK(cudaMemset(seen, 0, cap * 4)); CK(cudaMemset(popped, 0, 4));
  k_queue_init<<<1, 1>>>(D.S, cap, inq, pay); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  long long budget = budget_k * 1000;
  k_queue2<<<blocks, 32>>>(D.S, cap, budget, seed, inq, pay, pops, dbl, ab); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  k_drain<<<1, 32>>>(D.S, seen, cap, popped); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  long long p; int d, a, np; std::vector<unsigned> h(cap); std::vector<int> sn(cap);
  CK(cudaMemcpy(&p, pops, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&d, dbl, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&a, ab, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h.data(), pay, cap * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(sn.data(), seen, cap * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&np, popped, 4, cudaMemcpyDeviceToHost));
  long long sum = 0, lostq = 0, dupq = 0; for (int i = 0; i < cap; ++i) { sum += h[i]; if (sn[i] == 0) ++lostq; else if (sn[i] > 1) ++dupq; }
  long long lostupd = p - sum;
  char buf[96]; snprintf(buf, sizeof buf, "queue[cap=%d,ctas=%d]", cap, blocks);
  report(buf, p, d + (lostupd > 0 ? lostupd : -lostupd) + a + lostq + dupq + D.err(), "double_pop+lost_update+abort+drain_err+err");
  printf("       detail: pops=%lld (wraps/bucket ~%lld) double_pop=%d lost_updates=%lld aborts=%d drain_lost=%lld drain_dup=%lld err=%d\n", p, p / (cap ? cap : 1), d, lostupd, a, lostq, dupq, D.err());
}

int main(int argc, char** argv) {
  std::string t = argc > 1 ? argv[1] : "all"; int Rk = argc > 2 ? atoi(argv[2]) : 100, L = argc > 3 ? atoi(argv[3]) : 10; unsigned seed = argc > 4 ? (unsigned)atoi(argv[4]) : 1;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); printf("GPU %s sm_%d%d, %d SMs  [build:%s%s]\n", p.name, p.major, p.minor, p.multiProcessorCount,
#ifdef LIT_NOFENCE
    " NOFENCE"
#else
    " production header"
#endif
#ifdef LIT_MUTANT
    , " mutant=" LIT_MUTANT
#else
    , ""
#endif
  );
  bool all = t == "all";
  if (all || t == "sb") for (int v : {0, 1, 2, 3, 4}) run_sb(v, Rk, L, seed);
  if (all || t == "dekker1") run_dekker(1, Rk, L, seed);
  if (all || t == "dekker3") run_dekker(3, Rk / 2 > 0 ? Rk / 2 : 1, L, seed);
  if (all || t == "mp") for (int v : {0, 3, 1, 2}) run_mp(v, Rk > 200 ? 200 : Rk, L, seed);
  if (all || t == "queue") { for (int cap : {2, 4, 8, 64, 1024}) for (int b : {7, 200}) run_queue(cap, b, Rk * 20, seed); }
  printf("TOTAL rounds=%lld violations=%lld\n", g_total_rounds, g_total_bad);
  return g_total_bad ? 1 : 0;
}
