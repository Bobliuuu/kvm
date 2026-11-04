#pragma once
// KVM GPU executor: device runtime. One persistent kernel, block 0 = admitter, blocks 1.. = workers.
//
// ===== Roles and invariants (read this before changing anything) =====
//  * A worker block holds at most ONE task at a time and never waits on anything but the ready queue. A task whose
//    wait_var/wait_ver is unsatisfied is never in the queue: it is PARKED on its variable's lock-free list and
//    released by the publisher (or by its own parker's re-check). So no block can hold two tasks, and a publisher is
//    never blocked behind a waiter: waiters occupy no block and no queue cell.
//  * Every task is enqueued exactly once (last indegree decrementer, the admitter for roots, or the single winner of the
//    park CAS) and popped exactly once (head CAS); trace[slot].exec_count is the runtime witness.
//  * The ready queue has KVM_NBUCKETS FIFO rings (Vyukov bounded MPMC cells). Each holds at most live_cap entries
//    because the admitter admits a bundle only while (admitted - done + n) <= live_cap, so enqueue never finds a full
//    ring and never blocks on capacity. Rings wrap (a ring index is ticket & (live_cap-1)).
//  * Task slots are a bounded pool WITHOUT recycling: max_tasks per launch (also succ_cap/args_cap words). Exceeding it is
//    the explicit device error KVM_ERR_SLOTS/KVM_ERR_POOL, not UB. Only the admission ring and the ready rings recycle.
//  * Bundles share state only through version variables (no cross-bundle edges).
//
// ===== Memory-ordering argument for weight versions =====
//  Publisher P (task with pub_var=v, pub_ver=N):  all block threads store weights; __syncthreads(); thread 0 runs
//  __threadfence() (fence.sc.gpu, cumulative over what thread 0 has observed, which after bar.sync includes every
//  thread's stores); then CAS(version[v], N-1 -> N) with .release.gpu semantics.
//  Reader R (wait_var=v, wait_ver<=N):  the version is read with ld.acquire.gpu (in release_task before enqueue and again
//  in the worker right before the op). Reading N synchronizes-with P's release, so P's weight stores happen-before every
//  thread of R's block: R's thread 0 does the acquire, then __syncthreads() (bar.sync) orders it before the other threads'
//  loads (causality order is transitive through barriers). Hence a reader that sees version N sees all of version N's
//  weights (and anything older). Whether it can also see NEWER stores is the graph's job: an "exact" reader must be
//  ordered before the next writer by a version variable (reader publishes a "done" variable the writer waits on).
//  Task wakeups use acq_rel RMWs on indegree (release sequence over all predecessors) and release/acquire on queue cells,
//  so a successor sees the stores of all its predecessors.
//  Park/publish race (lost wakeup) is closed with a Dekker pattern: waiter = push; fence.sc; load version. publisher =
//  store version; fence.sc; exchange list. At least one side observes the other; the state CAS makes the release unique.
#include <cstdint>
#include <cuda_runtime.h>
#include "kvm_task_abi.h"

#define KVM_THREADS    256        // blockDim contract: all 256 threads enter every op (game ops: thread 0 = game thread)
#define KVM_NBUCKETS   32          // priority levels 0..31; larger runs first (validated, never silently clamped)
static_assert(KVM_NBUCKETS == 32, "kvm_pop maps one bucket to each lane of a warp");
#define KVM_SMEM_FLOATS 1024
#define KVM_MAX_ARGS   15
#define KVM_REC_MAGIC  0x4b564d52  // "KVMR"
#define KVM_REC_BUNDLE 0
#define KVM_REC_PAD    1
#define KVM_REC_HDR    8           // header words; all records are multiples of 8 words
#define KVM_TASK_WORDS 16

// ---------------- operator registry, generated from kvm_ops.def ----------------
enum KvmArgType { KVM_A_BUF = 0, KVM_A_I32 = 1, KVM_A_F32 = 2 };

namespace kvm_sig {
__host__ __device__ constexpr bool tok(const char* s, const char* w) {  // s starts with word w, then ',' or end
  while (*w) { if (*s != *w) return false; ++s; ++w; }
  return *s == ',' || *s == '\0';
}
__host__ __device__ constexpr int arity(const char* s) {
  if (!*s) return 0;
  int n = 1;
  for (; *s; ++s) if (*s == ',') ++n;
  return n;
}
// 2 bits per argument; 0xFFFFFFFF if the signature is malformed ("name:type,..." with type in buf|i32|f32)
__host__ __device__ constexpr uint32_t types(const char* s) {
  uint32_t m = 0;
  int i = 0;
  while (*s) {
    if (*s == ':' || *s == ',') return 0xFFFFFFFFu;  // empty argument name
    while (*s && *s != ':' && *s != ',') ++s;
    if (*s != ':') return 0xFFFFFFFFu;
    ++s;
    uint32_t t = tok(s, "buf") ? KVM_A_BUF : tok(s, "i32") ? KVM_A_I32 : tok(s, "f32") ? KVM_A_F32 : 3u;
    if (t == 3u || i >= KVM_MAX_ARGS) return 0xFFFFFFFFu;
    m |= t << (2 * i++);
    while (*s && *s != ',') ++s;
    if (*s == ',') ++s;
  }
  return m;
}
}  // namespace kvm_sig

enum KvmOpId {
#define X(n, id, sig) KVM_OP_##n = id,
#include "kvm_ops_list.inc"
#undef X
};

struct KvmOpInfo { const char* name; int id; int arity; uint32_t types; const char* sig; };
static constexpr KvmOpInfo kKvmOps[] = {
#define X(n, id, sig) {#n, id, kvm_sig::arity(sig), kvm_sig::types(sig), sig},
#include "kvm_ops_list.inc"
#undef X
};
static constexpr int KVM_NUM_OPS = sizeof(kKvmOps) / sizeof(kKvmOps[0]);

namespace kvm_sig {
constexpr bool registry_ok() {  // ids unique and >= 0, signatures well formed
  // KVM_DIST_HOOK: ids are sparse (comm ops live at 64..95), so uniqueness replaces "dense and equal to position".
  for (int i = 0; i < KVM_NUM_OPS; ++i) {
    if (kKvmOps[i].id < 0 || kKvmOps[i].types == 0xFFFFFFFFu) return false;
    for (int j = 0; j < i; ++j) if (kKvmOps[j].id == kKvmOps[i].id) return false;
  }
  return true;
}
}  // namespace kvm_sig
static_assert(kvm_sig::registry_ok(), "operator registry: ids must be unique and >= 0, every signature name:type with type buf|i32|f32");

// Compile-time view of the registry usable in device static_asserts (no device-side array odr-use).
__host__ __device__ constexpr int kvm_op_arity(int id) {
  switch (id) {
#define X(n, i, sig) case i: return kvm_sig::arity(sig);
#include "kvm_ops_list.inc"
#undef X
  }
  return -1;
}
__host__ __device__ constexpr int kvm_op_argtype(int id, int arg) {
  switch (id) {
#define X(n, i, sig) case i: return (int)((kvm_sig::types(sig) >> (2 * arg)) & 3u);
#include "kvm_ops_list.inc"
#undef X
  }
  return -1;
}

static const KvmOpInfo* kvm_op(int id) {  // nullptr if id is not registered
  for (const KvmOpInfo& o : kKvmOps) if (o.id == id) return &o;
  return nullptr;
}

// ---------------- errors ----------------
#define KVM_ERRORS(E) \
  E(OK, "ok") E(BAD_OP, "op_id not in registry") E(BAD_ARITY, "args_words != operator arity") \
  E(BAD_BUF_INDEX, "buffer index outside the device table") E(BAD_BUF, "buffer not f32 / ld != cols") \
  E(BOUNDS, "element count exceeds buffer") E(BAD_VAR, "version variable out of range") \
  E(VERSION_GAP, "pub_ver != version[pub_var]+1 (publisher not contiguous or two writers)") \
  E(WAIT_VIOLATION, "task ran with wait_var/wait_ver unsatisfied") E(DOUBLE_EXEC, "task executed twice") \
  E(BAD_RECORD, "malformed admission record") E(BAD_TASK, "task descriptor out of range (succ/args/priority/indegree)") \
  E(SLOTS, "task slot pool exhausted (max_tasks)") E(POOL, "successor/args pool exhausted") \
  E(BATCH_TOO_LARGE, "bundle larger than live_cap") E(TIMEOUT, "device timeout (deadlock or stalled admission)") \
  E(QUEUE_OVERFLOW, "ready ring invariant violated") E(BAD_CONFIG, "bad launch configuration")
enum KvmErr {
#define E(n, s) KVM_ERR_##n,
  KVM_ERRORS(E)
#undef E
};

// ---------------- shared host/device state ----------------
struct KvmTrace {  // one per slot, 64 B
  unsigned long long admit_ns, ready_ns, start_ns, end_ns;  // %globaltimer
  int smid, block, ver0, ver1;                              // ver0/ver1: wait_var version at start / end (-1 none)
  int flags, exec_count, pad0, pad1;                        // flags bit0 = deadline missed
};
#define KVM_TRACE_MISSED 1

struct KvmCell { unsigned long long seq, ready_ns; int slot, pad[3]; };            // 32 B
struct alignas(128) KvmQCtl { unsigned long long head, tail; };                    // own cache line

struct KvmCounters {  // device memory; hot fields on separate lines
  alignas(128) int done;
  alignas(128) int admitted, admit_final, slots_used, succ_used, args_used;
  alignas(128) int err_code, err_slot, err_a, err_b;
};
struct KvmHostErr { volatile int slot, a, b; volatile int code; };  // mapped mirror of the first device error

struct KvmDistDev;  // KVM_DIST_HOOK: include/kvm_dist.cuh
struct KvmDevState {
  int max_tasks, live_cap, succ_cap, args_cap, num_vars, nbufs, ring_words, pad;
  long long aging_ns;
  unsigned long long timeout_ns;
  const KvmBuf* bufs;
  KvmTaskDesc* tasks;
  int *indeg, *state, *park_next, *succ, *args, *version, *park_head;
  KvmTrace* trace;
  KvmCell* cells;
  KvmQCtl* qctl;
  KvmCounters* cnt;
  // host-visible (mapped pinned) control block
  int* ring;
  unsigned long long *doorbell, *ring_head;
  unsigned int* closed;
  KvmHostErr* herr;
  const KvmDistDev* dist;  // KVM_DIST_HOOK: nullptr = single-device executor (all dist code is skipped)
};

#ifdef __CUDACC__
// ---------------- PTX memory primitives ----------------
__device__ __forceinline__ unsigned long long kvm_now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int kvm_smid() { unsigned r; asm("mov.u32 %0, %%smid;" : "=r"(r)); return (int)r; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned long long ld_acq64(const unsigned long long* p) { unsigned long long v; asm volatile("ld.acquire.gpu.global.b64 %0,[%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rel64(unsigned long long* p, unsigned long long v) { asm volatile("st.release.gpu.global.b64 [%0],%1;" ::"l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ int atom_add(int* p, int v) { int r; asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],%2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
__device__ __forceinline__ unsigned long long atom_add64(unsigned long long* p, unsigned long long v) { unsigned long long r; asm volatile("atom.acq_rel.gpu.global.add.u64 %0,[%1],%2;" : "=l"(r) : "l"(p), "l"(v) : "memory"); return r; }
__device__ __forceinline__ int atom_cas(int* p, int c, int v) { int r; asm volatile("atom.acq_rel.gpu.global.cas.b32 %0,[%1],%2,%3;" : "=r"(r) : "l"(p), "r"(c), "r"(v) : "memory"); return r; }
__device__ __forceinline__ unsigned long long atom_cas64(unsigned long long* p, unsigned long long c, unsigned long long v) { unsigned long long r; asm volatile("atom.acq_rel.gpu.global.cas.b64 %0,[%1],%2,%3;" : "=l"(r) : "l"(p), "l"(c), "l"(v) : "memory"); return r; }
__device__ __forceinline__ int atom_exch(int* p, int v) { int r; asm volatile("atom.acq_rel.gpu.global.exch.b32 %0,[%1],%2;" : "=r"(r) : "l"(p), "r"(v) : "memory"); return r; }
// host-shared memory: system scope
__device__ __forceinline__ unsigned long long ld_acq_sys64(const unsigned long long* p) { unsigned long long v; asm volatile("ld.acquire.sys.global.b64 %0,[%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_acq_sys(const unsigned* p) { unsigned v; asm volatile("ld.acquire.sys.global.b32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ int ld_sys(const int* p) { int v; asm volatile("ld.relaxed.sys.global.b32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel_sys64(unsigned long long* p, unsigned long long v) { asm volatile("st.release.sys.global.b64 [%0],%1;" ::"l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ void st_rel_sys(volatile int* p, int v) { asm volatile("st.release.sys.global.b32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }

// ---------------- errors ----------------
// First error wins (CAS in device memory), is mirrored to the host-mapped block, and makes every block exit.
__device__ __noinline__ void kvm_fail(KvmCounters* cnt, KvmHostErr* herr, int code, int slot, int a, int b) {
  if (atomicCAS(&cnt->err_code, 0, code) != 0) return;
  cnt->err_slot = slot; cnt->err_a = a; cnt->err_b = b;
  herr->slot = slot; herr->a = a; herr->b = b;
  st_rel_sys(&herr->code, code);
}
__device__ __forceinline__ void kvm_fail(const KvmDevState& S, int code, int slot, int a = 0, int b = 0) { kvm_fail(S.cnt, S.herr, code, slot, a, b); }

// ---------------- operator context and typed arguments ----------------
struct KvmView { float* p; int rows, cols; };

struct KvmCtx {
  const KvmBuf* bufs;
  int nbufs, slot;
  KvmCounters* cnt;
  KvmHostErr* herr;
  float* smem;  // KVM_SMEM_FLOATS floats, block-shared scratch
  __device__ void fail(int code, int a = 0, int b = 0) const { kvm_fail(cnt, herr, code, slot, a, b); }
  // Typed accessors. OP/I are checked against kvm_ops.def at compile time (see KVM_OP): a device function that reads an
  // argument as the wrong type, or past the registered arity, does not compile.
  template <int OP, int I> __device__ KvmView buf(const int32_t* a) const {
    static_assert(I < kvm_op_arity(OP) && kvm_op_argtype(OP, I) == KVM_A_BUF, "argument is not a buf in kvm_ops.def");
    int idx = a[I];
    if (idx < 0 || idx >= nbufs) { fail(KVM_ERR_BAD_BUF_INDEX, idx, I); return {nullptr, 0, 0}; }
    const KvmBuf& b = bufs[idx];
    if (b.dtype != 0 || b.ld != b.cols || b.p == nullptr) { fail(KVM_ERR_BAD_BUF, idx, I); return {nullptr, 0, 0}; }
    return {(float*)b.p, b.rows, b.cols};
  }
  template <int OP, int I> __device__ int i32(const int32_t* a) const {
    static_assert(I < kvm_op_arity(OP) && kvm_op_argtype(OP, I) == KVM_A_I32, "argument is not an i32 in kvm_ops.def");
    return a[I];
  }
  template <int OP, int I> __device__ float f32(const int32_t* a) const {
    static_assert(I < kvm_op_arity(OP) && kvm_op_argtype(OP, I) == KVM_A_F32, "argument is not an f32 in kvm_ops.def");
    return __int_as_float(a[I]);
  }
  // 0 <= n <= elements of every view; false (after reporting) otherwise. An invalid view was already reported by buf().
  template <class... V> __device__ bool fits(int n, V... v) const {
    if ((... || (v.p == nullptr))) return false;
    bool ok = n >= 0;
    ((ok = ok && (long long)n <= (long long)v.rows * v.cols), ...);
    if (!ok) fail(KVM_ERR_BOUNDS, n);
    return ok;
  }
};

#define KVM_OP(n) template <int KVM_SELF = KVM_OP_##n> __device__ __forceinline__ void op_##n(const KvmCtx& c, const int32_t* a)
#define ARG_BUF(i) c.template buf<KVM_SELF, (i)>(a)
#define ARG_I32(i) c.template i32<KVM_SELF, (i)>(a)
#define ARG_F32(i) c.template f32<KVM_SELF, (i)>(a)

// ---------------- operators (block-cooperative: every thread of the block calls with the same args) ----------------
KVM_OP(noop) {}
KVM_OP(copy) {
  KvmView s = ARG_BUF(0), d = ARG_BUF(1); int n = ARG_I32(2);
  if (!c.fits(n, s, d)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) d.p[i] = s.p[i];
}
KVM_OP(add) {
  KvmView x = ARG_BUF(0), y = ARG_BUF(1), o = ARG_BUF(2); int n = ARG_I32(3);
  if (!c.fits(n, x, y, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = x.p[i] + y.p[i];
}
KVM_OP(scale) {
  KvmView x = ARG_BUF(0), o = ARG_BUF(1); float al = ARG_F32(2); int n = ARG_I32(3);
  if (!c.fits(n, x, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = al * x.p[i];
}
KVM_OP(relu) {
  KvmView x = ARG_BUF(0), o = ARG_BUF(1); int n = ARG_I32(2);
  if (!c.fits(n, x, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = x.p[i] > 0.f ? x.p[i] : 0.f;
}
KVM_OP(sgd) {
  KvmView w = ARG_BUF(0), g = ARG_BUF(1); float lr = ARG_F32(2); int n = ARG_I32(3);
  if (!c.fits(n, w, g)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) w.p[i] = w.p[i] - lr * g.p[i];
}
KVM_OP(predict) {
  KvmView x = ARG_BUF(0), w = ARG_BUF(1), o = ARG_BUF(2); int n = ARG_I32(3);
  if (!c.fits(n, x, o) || !c.fits(1, w)) return;
  float w0 = w.p[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = w0 * x.p[i];
}
// Deterministic reduction order (the CPU reference in tests replicates it): thread t sums elements t, t+256, ... in order;
// then a halving tree over the 256 partials.
KVM_OP(mse_grad) {
  KvmView p = ARG_BUF(0), t = ARG_BUF(1), x = ARG_BUF(2), loss = ARG_BUF(3), grad = ARG_BUF(4); int n = ARG_I32(5);
  if (n < 1) { c.fail(KVM_ERR_BOUNDS, n); return; }
  if (!c.fits(n, p, t, x) || !c.fits(1, loss, grad)) return;
  float* sl = c.smem;
  float* sg = c.smem + KVM_THREADS;
  float la = 0.f, ga = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float d = p.p[i] - t.p[i];
    la += d * d;
    ga += (2.f * d) * x.p[i];
  }
  sl[threadIdx.x] = la; sg[threadIdx.x] = ga;
  __syncthreads();
  for (int s = KVM_THREADS / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) { sl[threadIdx.x] += sl[threadIdx.x + s]; sg[threadIdx.x] += sg[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { loss.p[0] = sl[0] / (float)n; grad.p[0] = sg[0] / (float)n; }
}
// fp32, 16x16 smem tiles, any m,n,k. Each output element accumulates k ascending from 0 (same order as a naive loop).
KVM_OP(matmul) {
  KvmView A = ARG_BUF(0), B = ARG_BUF(1), O = ARG_BUF(2); int m = ARG_I32(3), n = ARG_I32(4), k = ARG_I32(5);
  if (A.p == nullptr || B.p == nullptr || O.p == nullptr) return;
  if (m < 1 || n < 1 || k < 1 || m > A.rows || k > A.cols || k > B.rows || n > B.cols || m > O.rows || n > O.cols) {
    c.fail(KVM_ERR_BOUNDS, m, n); return;
  }
  float* As = c.smem;             // 16x16
  float* Bs = c.smem + 256;       // 16x16
  int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  for (int bm = 0; bm < m; bm += 16)
    for (int bn = 0; bn < n; bn += 16) {
      float acc = 0.f;
      for (int bk = 0; bk < k; bk += 16) {
        int ar = bm + ty, ac = bk + tx, br = bk + ty, bc = bn + tx;
        As[ty * 16 + tx] = (ar < m && ac < k) ? A.p[(size_t)ar * A.cols + ac] : 0.f;
        Bs[ty * 16 + tx] = (br < k && bc < n) ? B.p[(size_t)br * B.cols + bc] : 0.f;
        __syncthreads();
        int kmax = min(16, k - bk);
        for (int kk = 0; kk < kmax; ++kk) acc += As[ty * 16 + kk] * Bs[kk * 16 + tx];
        __syncthreads();
      }
      if (bm + ty < m && bn + tx < n) O.p[(size_t)(bm + ty) * O.cols + bn + tx] = acc;
    }
}
KVM_OP(fill) {
  KvmView x = ARG_BUF(0); float v = ARG_F32(1); int n = ARG_I32(2);
  if (!c.fits(n, x)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) x.p[i] = v;
}
KVM_OP(spin) {
  int us = ARG_I32(0);
  if (threadIdx.x == 0) { unsigned long long end = kvm_now() + (unsigned long long)(us < 0 ? 0 : us) * 1000ull; while (kvm_now() < end) {} }
}

// Optional operator packs: each defines KVM_OP(name) for every X line of its .def.
#include "kvm_dist.cuh"  // KVM_DIST_HOOK: phase 1 (types, primitives); phase 2 (inbox pump) follows kvm_publish
#if __has_include("kvm_ops_comm.cuh")
#include "kvm_ops_comm.cuh"  // KVM_DIST_HOOK
#endif
#if __has_include("kvm_ops_gpt.cuh")
#include "kvm_ops_gpt.cuh"
#endif
#if __has_include("kvm_ops_game.cuh")
#include "kvm_ops_game.cuh"
#endif

// Adding an operator = one X line in kvm_ops.def + one KVM_OP(name) above; a missing KVM_OP is a compile error here.
__device__ __forceinline__ void kvm_dispatch(int op, const KvmCtx& c, const int32_t* a) {
  switch (op) {
#define X(n, id, sig) case id: op_##n(c, a); break;
#include "kvm_ops_list.inc"
#undef X
    default: c.fail(KVM_ERR_BAD_OP, op);
  }
}

// Device-side registry lookups (admission validation): -1 / 0 for an unregistered id.
__device__ __forceinline__ int kvm_dev_arity(int op) { return kvm_op_arity(op); }
__device__ __forceinline__ uint32_t kvm_dev_types(int op) {
  switch (op) {
#define X(n, id, sig) case id: return kvm_sig::types(sig);
#include "kvm_ops_list.inc"
#undef X
  }
  return 0;
}

// ---------------- ready queue ----------------
// bucket b = priority (validated 0..31). A pop picks, over the head of every bucket, the max of
//   score = b * aging_ns + (now - ready_ns)        (aging_ns = 1<<50 when aging is off: strict priority)
// Within a bucket the head is the oldest, so this is exact "priority + aging" over all ready tasks; FIFO within bucket.
__device__ __forceinline__ void kvm_enqueue(const KvmDevState& S, int slot) {
  int b = S.tasks[slot].priority;
  KvmQCtl* q = S.qctl + b;
  unsigned long long pos = atom_add64(&q->tail, 1ull);
  KvmCell* cell = S.cells + (size_t)b * S.live_cap + (pos & (unsigned long long)(S.live_cap - 1));
  // The cell frees when its previous occupant was popped; the occupant's popper recycles it right after its head CAS.
  for (int spin = 0; ld_acq64(&cell->seq) != pos; ++spin) {
    if (spin > (1 << 22)) { kvm_fail(S, KVM_ERR_QUEUE_OVERFLOW, slot, b, (int)pos); return; }
    __nanosleep(32);
  }
  unsigned long long now = kvm_now();
  S.trace[slot].ready_ns = now;
  cell->slot = slot;
  cell->ready_ns = now;
  st_rel64(&cell->seq, pos + 1);
}

// Warp-wide (all 32 lanes call). Returns a slot, or -1 if nothing was claimed.
__device__ __forceinline__ int kvm_pop(const KvmDevState& S, unsigned long long now) {
  const int lane = threadIdx.x & 31;
  for (int attempt = 0; attempt < 4; ++attempt) {
    KvmQCtl* q = S.qctl + lane;
    unsigned long long h = ld_acq64(&q->head);
    KvmCell* cell = S.cells + (size_t)lane * S.live_cap + (h & (unsigned long long)(S.live_cap - 1));
    long long score = -1;
    if (ld_acq64(&cell->seq) == h + 1) {
      long long age = (long long)(now - cell->ready_ns);
      score = (long long)lane * S.aging_ns + (age > 0 ? age : 0);
    }
    long long best = score; int bl = lane;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      long long o = __shfl_xor_sync(0xffffffffu, best, off);
      int ol = __shfl_xor_sync(0xffffffffu, bl, off);
      if (o > best || (o == best && ol > bl)) { best = o; bl = ol; }
    }
    if (best < 0) return -1;
    int got = -2;
    if (lane == bl) {
      if (atom_cas64(&q->head, h, h + 1) == h) {
        got = cell->slot;
        st_rel64(&cell->seq, h + (unsigned long long)S.live_cap);  // recycle before running
      }
    }
    got = __shfl_sync(0xffffffffu, got, bl);
    if (got >= 0) return got;
  }
  return -1;
}

// ---------------- release: indegree reached 0 ----------------
// Enqueue now if the version wait is satisfied, else park (see header comment for the lost-wakeup argument).
__device__ __forceinline__ void kvm_park(const KvmDevState& S, int slot, int var, int need);
__device__ __forceinline__ void kvm_release_task(const KvmDevState& S, int slot) {
  const KvmTaskDesc* d = S.tasks + slot;
  int var = d->wait_var;
  if (var < 0 || ld_acq(S.version + var) >= d->wait_ver) kvm_enqueue(S, slot);
  else { S.state[slot] = 1; kvm_park(S, slot, var, d->wait_ver); }
}
__device__ __forceinline__ void kvm_park(const KvmDevState& S, int slot, int var, int need) {
  int* head = S.park_head + var;
  int old;
  do { old = *(volatile int*)head; S.park_next[slot] = old; } while (atom_cas(head, old, slot) != old);
  __threadfence();  // seq_cst fence: push -> fence -> load version   (publisher: store version -> fence -> exchange)
  if (ld_acq(S.version + var) >= need && atom_cas(S.state + slot, 1, 2) == 1) kvm_enqueue(S, slot);
}

// version[pub_var]: N-1 -> N. Caller has passed a block barrier after the op's stores.
__device__ __forceinline__ void kvm_publish(const KvmDevState& S, int slot, int var, int ver) {
  __threadfence();                                              // fence.sc.gpu: weights before version (see header)
  int old = atom_cas(S.version + var, ver - 1, ver);            // release store, and detects gaps / double writers
  if (old != ver - 1) { kvm_fail(S, KVM_ERR_VERSION_GAP, slot, old, ver); return; }
  __threadfence();                                              // version store -> fence -> list exchange
  int n = atom_exch(S.park_head + var, -1);
  while (n >= 0) {
    int next = S.park_next[n];                                  // read before a re-park rewrites it
    int need = S.tasks[n].wait_ver;
    if (ld_acq(S.version + var) >= need) { if (atom_cas(S.state + n, 1, 2) == 1) kvm_enqueue(S, n); }
    else kvm_park(S, n, var, need);
    n = next;
  }
}

#define KVM_DIST_PUMP_PHASE 1  // KVM_DIST_HOOK: remote-publish pump (kvm_dist_poll / kvm_dist_pump)
#include "kvm_dist.cuh"

// ---------------- worker ----------------
struct KvmShared {
  KvmTaskDesc d;
  int cmd, skip;
  float smem[KVM_SMEM_FLOATS];
};
#define KVM_POP_EXIT (-2)

__device__ __forceinline__ bool kvm_should_exit(const KvmDevState& S, unsigned long long t0) {
  if (*(volatile int*)&S.cnt->err_code) return true;
  if (kvm_now() - t0 > S.timeout_ns) {
    kvm_fail(S, KVM_ERR_TIMEOUT, -1, ld_acq(&S.cnt->admitted), ld_acq(&S.cnt->done));
    return true;
  }
  if (ld_acq(&S.cnt->admit_final)) return ld_acq(&S.cnt->done) >= ld_acq(&S.cnt->admitted);
  return false;
}

__device__ void kvm_run_task(const KvmDevState& S, int slot, KvmShared& sh) {
  if (threadIdx.x < KVM_TASK_WORDS) ((int*)&sh.d)[threadIdx.x] = ((const int*)(S.tasks + slot))[threadIdx.x];
  __syncthreads();
  const KvmTaskDesc& d = sh.d;
  KvmTrace* tr = S.trace + slot;
  if (threadIdx.x == 0) {
    bool bad = false;
    tr->start_ns = kvm_now(); tr->smid = kvm_smid(); tr->block = blockIdx.x; tr->ver0 = tr->ver1 = -1;
    if (atomicAdd(&tr->exec_count, 1) != 0) { kvm_fail(S, KVM_ERR_DOUBLE_EXEC, slot); bad = true; }
    if (d.wait_var >= 0) {
      int v = ld_acq(S.version + d.wait_var);  // acquire: pairs with the publisher's release (header comment)
      tr->ver0 = v;
      if (v < d.wait_ver) { kvm_fail(S, KVM_ERR_WAIT_VIOLATION, slot, v, d.wait_ver); bad = true; }
    }
#ifdef KVM_BREAK_FENCE  // negative test: publish BEFORE the op's stores (no happens-before) so version tests must fail
    if (d.pub_var >= 0) kvm_publish(S, slot, d.pub_var, d.pub_ver);
#endif
    sh.skip = bad || *(volatile int*)&S.cnt->err_code;
  }
  __syncthreads();
  if (!sh.skip) {
    KvmCtx c{S.bufs, S.nbufs, slot, S.cnt, S.herr, sh.smem};
    kvm_dispatch(d.op_id, c, S.args + d.args_off);
  }
  __syncthreads();  // all op stores of the block precede thread 0's fence below
  if (threadIdx.x == 0) {
    unsigned long long end = kvm_now();
    tr->end_ns = end;
    if (d.wait_var >= 0) tr->ver1 = ld_acq(S.version + d.wait_var);
    if (d.deadline_us >= 0 && end - tr->admit_ns > (unsigned long long)d.deadline_us * 1000ull) tr->flags |= KVM_TRACE_MISSED;
#ifndef KVM_BREAK_FENCE
    if (d.pub_var >= 0 && !sh.skip) kvm_publish(S, slot, d.pub_var, d.pub_ver);  // publish THEN wake successors
#endif
  }
  __syncthreads();
  if (!sh.skip) {
    __threadfence();
    for (int e = threadIdx.x; e < d.succ_cnt; e += blockDim.x) {
      int s = S.succ[d.succ_off + e];
      if (atom_add(S.indeg + s, -1) == 1) kvm_release_task(S, s);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) atom_add(&S.cnt->done, 1);
}

__device__ void kvm_worker(const KvmDevState& S, unsigned long long t0, KvmShared& sh) {
  unsigned backoff = 0;
  for (;;) {
    if (threadIdx.x < 32) {
      int r = kvm_pop(S, kvm_now());
      if (r < 0 && threadIdx.x == 0 && kvm_should_exit(S, t0)) r = KVM_POP_EXIT;
      if (threadIdx.x == 0) sh.cmd = r;
    }
    __syncthreads();
    int cmd = sh.cmd;
    __syncthreads();  // sh.cmd is rewritten next iteration
    if (cmd == KVM_POP_EXIT) return;
    if (cmd < 0) {
      if (threadIdx.x < 32) __nanosleep(backoff);
      backoff = backoff ? min(backoff * 2, 512u) : 32u;
      continue;
    }
    backoff = 0;
    kvm_run_task(S, cmd, sh);
  }
}

// ---------------- admitter (block 0) ----------------
struct KvmAdmitShared { int action, words, hdr[KVM_REC_HDR]; int base, sbase, abase; };
#define KVM_ADM_IDLE 0
#define KVM_ADM_REC  1
#define KVM_ADM_FINAL 2
#define KVM_ADM_EXIT 3

__device__ void kvm_admit_record(const KvmDevState& S, unsigned long long t0, KvmAdmitShared& A, unsigned long long pos) {
  const int* rec = S.ring + (size_t)(pos % (unsigned long long)S.ring_words);
  const int nt = A.hdr[2], ne = A.hdr[3], na = A.hdr[4];
  const int* rt = rec + KVM_REC_HDR;
  const int* re = rt + (size_t)nt * KVM_TASK_WORDS;
  const int* ra = re + ne;
  if (threadIdx.x == 0) {
    KvmCounters* c = S.cnt;
    int code = 0;
    if (nt < 1 || ne < 0 || na < 0) code = KVM_ERR_BAD_RECORD;
    else if (nt > S.live_cap) code = KVM_ERR_BATCH_TOO_LARGE;
    else if (c->slots_used + nt > S.max_tasks) code = KVM_ERR_SLOTS;
    else if (c->succ_used + ne > S.succ_cap || c->args_used + na > S.args_cap) code = KVM_ERR_POOL;
    if (code) { kvm_fail(S, code, -1, nt, c->slots_used); A.action = KVM_ADM_EXIT; }
    else {
      // backpressure: wait until the live window (admitted - done) has room; a stall here is reported by the timeout
      while (ld_acq(&c->admitted) - ld_acq(&c->done) + nt > S.live_cap) {
        if (*(volatile int*)&c->err_code || kvm_now() - t0 > S.timeout_ns) { A.action = KVM_ADM_EXIT; break; }
        __nanosleep(256);
      }
      if (A.action != KVM_ADM_EXIT) { A.base = c->slots_used; A.sbase = c->succ_used; A.abase = c->args_used; }
    }
  }
  __syncthreads();
  if (A.action == KVM_ADM_EXIT) return;
  const unsigned long long now = kvm_now();
  for (int i = threadIdx.x; i < nt; i += blockDim.x) {
    int w[KVM_TASK_WORDS];
#pragma unroll
    for (int j = 0; j < KVM_TASK_WORDS; ++j) w[j] = ld_sys(rt + (size_t)i * KVM_TASK_WORDS + j);
    // KvmTaskDesc word order: op, succ_off, succ_cnt, indegree, priority, deadline, args_off, args_words, wait_var,
    // wait_ver, pub_var, pub_ver, workload, flags, r0, r1
    int slot = A.base + i;
    int op = w[0], so = w[1], sc = w[2], ind = w[3], pr = w[4], ao = w[6], aw = w[7], wv = w[8], pv = w[10];
    int code = 0;
    if (kvm_dev_arity(op) < 0) code = KVM_ERR_BAD_OP;
    else if (aw != kvm_dev_arity(op)) code = KVM_ERR_BAD_ARITY;
    else if (so < 0 || sc < 0 || so + sc > ne || ao < 0 || aw < 0 || ao + aw > na || ind < 0 || ind > nt || pr < 0 || pr >= KVM_NBUCKETS || w[5] < -1) code = KVM_ERR_BAD_TASK;
    else if (wv < -1 || wv >= S.num_vars || pv < -1 || pv >= S.num_vars) code = KVM_ERR_BAD_VAR;
    else {
      uint32_t ty = kvm_dev_types(op);
      for (int j = 0; j < aw && !code; ++j)
        if (((ty >> (2 * j)) & 3u) == KVM_A_BUF) { int bi = ld_sys(ra + ao + j); if (bi < 0 || bi >= S.nbufs) code = KVM_ERR_BAD_BUF_INDEX; }
    }
    if (code) { kvm_fail(S, code, slot, op, i); continue; }
    w[1] = so + A.sbase;  // rebase CSR offsets to the global pools
    w[6] = ao + A.abase;
    int* dst = (int*)(S.tasks + slot);
#pragma unroll
    for (int j = 0; j < KVM_TASK_WORDS; ++j) dst[j] = w[j];
    S.indeg[slot] = ind; S.state[slot] = 0; S.park_next[slot] = -1;
    KvmTrace& t = S.trace[slot];
    t.admit_ns = now; t.ready_ns = t.start_ns = t.end_ns = 0; t.flags = 0; t.exec_count = 0; t.ver0 = t.ver1 = -1;
  }
  for (int e = threadIdx.x; e < ne; e += blockDim.x) {
    int v = ld_sys(re + e);
    if (v < 0 || v >= nt) { kvm_fail(S, KVM_ERR_BAD_TASK, A.base, v, e); v = 0; }
    S.succ[A.sbase + e] = A.base + v;
  }
  for (int j = threadIdx.x; j < na; j += blockDim.x) S.args[A.abase + j] = ld_sys(ra + j);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!*(volatile int*)&S.cnt->err_code) {
      __threadfence();
      S.cnt->slots_used = A.base + nt; S.cnt->succ_used = A.sbase + ne; S.cnt->args_used = A.abase + na;
      atom_add(&S.cnt->admitted, nt);  // before any task of this bundle can be enqueued (done <= admitted always)
    }
  }
  __syncthreads();
  if (*(volatile int*)&S.cnt->err_code) return;
  for (int i = threadIdx.x; i < nt; i += blockDim.x)
    if (S.tasks[A.base + i].indegree == 0) kvm_release_task(S, A.base + i);  // the descriptor is immutable; S.indeg is live and may already be 0 for a non-root
}

__device__ void kvm_admitter(const KvmDevState& S, unsigned long long t0, KvmAdmitShared& A) {
  unsigned long long pos = 0;
  for (;;) {
    if (threadIdx.x == 0) {
      A.action = KVM_ADM_IDLE;
      if (*(volatile int*)&S.cnt->err_code || kvm_now() - t0 > S.timeout_ns) {
        kvm_fail(S, KVM_ERR_TIMEOUT, -1, ld_acq(&S.cnt->admitted), ld_acq(&S.cnt->done));
        A.action = KVM_ADM_EXIT;
      } else {
        unsigned closed = ld_acq_sys(S.closed);                    // BEFORE the doorbell: closed => doorbell is final
        unsigned long long tail = ld_acq_sys64(S.doorbell);        // acquire: record words below are visible
        if (pos < tail) {
          const int* rec = S.ring + (size_t)(pos % (unsigned long long)S.ring_words);
          for (int j = 0; j < KVM_REC_HDR; ++j) A.hdr[j] = ld_sys(rec + j);
          unsigned long long left = (unsigned long long)S.ring_words - pos % (unsigned long long)S.ring_words;
          bool ok = A.hdr[0] == KVM_REC_MAGIC && A.hdr[1] >= KVM_REC_HDR && A.hdr[1] % 8 == 0 &&
                    (unsigned long long)A.hdr[1] <= left && pos + A.hdr[1] <= tail &&
                    (A.hdr[6] == KVM_REC_PAD || (A.hdr[6] == KVM_REC_BUNDLE && A.hdr[1] == ((KVM_REC_HDR + A.hdr[2] * KVM_TASK_WORDS + A.hdr[3] + A.hdr[4] + 7) & ~7)));
          if (!ok) { kvm_fail(S, KVM_ERR_BAD_RECORD, -1, (int)pos, A.hdr[1]); A.action = KVM_ADM_EXIT; }
          else A.action = KVM_ADM_REC;
        } else if (closed) A.action = KVM_ADM_FINAL;
      }
    }
    __syncthreads();
    int action = A.action;
    __syncthreads();
    if (action == KVM_ADM_EXIT) return;
    if (action == KVM_ADM_FINAL) { if (threadIdx.x == 0) st_rel(&S.cnt->admit_final, 1); kvm_dist_pump(S, t0); return; }  // KVM_DIST_HOOK
    if (action == KVM_ADM_IDLE) { kvm_dist_poll(S); __nanosleep(256); continue; }  // KVM_DIST_HOOK
    if (A.hdr[6] == KVM_REC_BUNDLE) kvm_admit_record(S, t0, A, pos);
    __syncthreads();
    if (A.action == KVM_ADM_EXIT) return;
    pos += (unsigned long long)A.hdr[1];
    if (threadIdx.x == 0) st_rel_sys64(S.ring_head, pos);  // ring space returns to the host only after the copy
    __syncthreads();
  }
}

__global__ void __launch_bounds__(KVM_THREADS) kvm_init_kernel(KvmDevState S) {
  size_t n = (size_t)KVM_NBUCKETS * S.live_cap;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    S.cells[i].seq = i % (size_t)S.live_cap; S.cells[i].slot = -1;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < S.num_vars; i += gridDim.x * blockDim.x) S.park_head[i] = -1;
}

__global__ void __launch_bounds__(KVM_THREADS) kvm_exec_kernel(const __grid_constant__ KvmDevState S) {
  __shared__ KvmShared sh;
  __shared__ KvmAdmitShared adm;
  const unsigned long long t0 = kvm_now();
  if (gridDim.x < 2 || (S.live_cap & (S.live_cap - 1)) || S.live_cap < 2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) kvm_fail(S, KVM_ERR_BAD_CONFIG, -1, gridDim.x, S.live_cap);
    return;
  }
  if (blockIdx.x == 0) kvm_admitter(S, t0, adm);
  else kvm_worker(S, t0, sh);
}
#endif  // __CUDACC__
