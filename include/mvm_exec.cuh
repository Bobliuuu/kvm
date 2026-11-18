#pragma once
// Magni VM GPU executor: device runtime. One persistent kernel, block 0 = admitter, blocks 1.. = workers.
//
// ===== Roles and invariants (read this before changing anything) =====
//  * A worker block holds at most ONE task at a time and never waits on anything but the ready queue. A task whose
//    wait_var/wait_ver is unsatisfied is never in the queue: it is PARKED on its variable's lock-free list and
//    released by the publisher (or by its own parker's re-check). So no block can hold two tasks, and a publisher is
//    never blocked behind a waiter: waiters occupy no block and no queue cell.
//  * Every task is enqueued exactly once (last indegree decrementer, the admitter for roots, or the single winner of the
//    park CAS) and popped exactly once (head CAS); trace[slot].exec_count is the runtime witness.
//  * The ready queue has MVM_NBUCKETS FIFO rings per class (Vyukov bounded MPMC cells), each rcap = 4 * live_cap deep. Two budgets keep
//    every ring from filling: admission waits while (admitted - done - parked + n) > live_cap (the runnable window) or
//    parked > live_cap (version-waiting tasks have their own budget, so a bundle that publishes a version is never blocked behind
//    waiters of an earlier bundle), hence at most 2 * live_cap tasks are ever in flight (a record is admitted only into room in the
//    runnable window, its n tasks included, while at most live_cap are parked; parking moves a task between the two) and enqueue never
//    finds a full ring. More than live_cap parked at once is the explicit error PARK_STALL. Rings wrap (index = ticket & (rcap-1)).
//  * `timeout` is a STALL detector, not a lifetime cap: pending work and no task finished or admitted for that long. Idle waiting
//    for the host never counts. It must exceed the longest task; task_limit reports (after the fact) a task that ran longer.
//  * Task slots are a bounded pool WITHOUT recycling: max_tasks per launch (also succ_cap/args_cap words). Exceeding it is
//    the explicit device error MVM_ERR_SLOTS/MVM_ERR_POOL, not UB. Only the admission ring and the ready rings recycle.
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
#include "mvm_task_abi.h"
#include "mvm_sched.h"      // SCHED: scheduling policy data (host+device); policy code is include/mvm_sched.cuh

#ifndef MVM_MIN_BLOCKS
#define MVM_MIN_BLOCKS 1          // resident blocks per SM the kernel is compiled for (register cap = 65536 / (MVM_THREADS * MVM_MIN_BLOCKS))
#endif
#define MVM_THREADS    256        // blockDim contract: all 256 threads enter every op (game ops: thread 0 = game thread)
#ifndef MVM_MIN_BLOCKS
#define MVM_MIN_BLOCKS 1          // __launch_bounds__ minimum resident blocks per SM. 2 caps the persistent kernel at 128 registers (65536 / 512) so one
                                  // block's stalls overlap the other's (+40-60% on matmul, +25% attention backward in the op harness), but the executor suites
                                  // have NOT been run at 2 yet (results/2026-11-06-matmul2-perf): build with -DMVM_MIN_BLOCKS=2 to try it. 1 = 255 registers.
#endif
#define MVM_NBUCKETS   32          // priority levels 0..31; larger runs first (validated, never silently clamped)
static_assert(MVM_NBUCKETS == 32, "mvm_pop maps one bucket to each lane of a warp");
#define MVM_SMEM_FLOATS 512     // the block's float scratch (MvmCtx::smem). Its users need two MVM_THREADS-float halves (the loss reduction, the
                                // 16x16 matmul's tiles). At 1024 the executor's static shared memory put a 96 KiB launch 1,792 B over the A100's
                                // 100 KB carveout (98,304 + 4,864 + the 1 KB the block reserves), so it ran with 60 KB of L1 instead of 92.
static_assert(2 * MVM_THREADS <= MVM_SMEM_FLOATS, "MvmCtx::smem holds two MVM_THREADS-float halves");
#define MVM_PG_SUBS 6              // MVM_PROF_GAP: sub-steps a phase records (they go to gp[9..14])
#define MVM_MAX_ARGS   20          // arguments per operator (3 bits each in a 64-bit type mask); slots hold MVM_ARG_STRIDE words
#define MVM_REC_MAGIC  0x4b564d52  // "MVMR"
#define MVM_REC_BUNDLE 0
#define MVM_REC_PAD    1
#define MVM_REC_HDR    8           // header words; all records are multiples of 8 words
#define MVM_TASK_WORDS 16

// ---------------- operator registry, generated from mvm_ops.def ----------------
enum MvmArgType { MVM_A_BUF = 0, MVM_A_I32 = 1, MVM_A_F32 = 2, MVM_A_BUF_OPT = 3 };  // buf? : word -1 = absent

namespace mvm_sig {
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
constexpr uint64_t MVM_BAD_SIG = ~0ull;
// 3 bits per argument; MVM_BAD_SIG if malformed ("name:type,..." with type in buf|buf?|i32|f32)
__host__ __device__ constexpr uint64_t types(const char* s) {
  uint64_t m = 0;
  int i = 0;
  while (*s) {
    if (*s == ':' || *s == ',') return MVM_BAD_SIG;  // empty argument name
    while (*s && *s != ':' && *s != ',') ++s;
    if (*s != ':') return MVM_BAD_SIG;
    ++s;
    int t = tok(s, "buf") ? MVM_A_BUF : tok(s, "buf?") ? MVM_A_BUF_OPT : tok(s, "i32") ? MVM_A_I32 : tok(s, "f32") ? MVM_A_F32 : -1;
    if (t < 0 || i >= MVM_MAX_ARGS) return MVM_BAD_SIG;
    m |= (uint64_t)t << (3 * i++);
    while (*s && *s != ',') ++s;
    if (*s == ',') ++s;
  }
  return m;
}
}  // namespace mvm_sig

enum MvmOpId {
#define X(n, id, sig) MVM_OP_##n = id,
#include "mvm_ops_list.inc"
#undef X
};

struct MvmOpInfo { const char* name; int id; int arity; uint64_t types; const char* sig; };
static constexpr MvmOpInfo kMvmOps[] = {
#define X(n, id, sig) {#n, id, mvm_sig::arity(sig), mvm_sig::types(sig), sig},
#include "mvm_ops_list.inc"
#undef X
};
static constexpr int MVM_NUM_OPS = sizeof(kMvmOps) / sizeof(kMvmOps[0]);

namespace mvm_sig {
constexpr bool registry_ok() {  // MVM_DIST_HOOK: ids unique and >= 0 (sparse across the .def packs), signatures well formed
  for (int i = 0; i < MVM_NUM_OPS; ++i) {
    if (kMvmOps[i].id < 0 || kMvmOps[i].types == mvm_sig::MVM_BAD_SIG) return false;
    for (int j = 0; j < i; ++j) if (kMvmOps[j].id == kMvmOps[i].id) return false;
  }
  return true;
}
}  // namespace mvm_sig
static_assert(mvm_sig::registry_ok(), "operator registry: ids must be unique and >= 0, every signature name:type with type buf|buf?|i32|f32");

// Compile-time view of the registry usable in device static_asserts (no device-side array odr-use).
__host__ __device__ constexpr int mvm_op_arity(int id) {
  switch (id) {
#define X(n, i, sig) case i: return mvm_sig::arity(sig);
#include "mvm_ops_list.inc"
#undef X
  }
  return -1;
}
__host__ __device__ constexpr int mvm_op_argtype(int id, int arg) {
  switch (id) {
#define X(n, i, sig) case i: return (int)((mvm_sig::types(sig) >> (3 * arg)) & 7u);
#include "mvm_ops_list.inc"
#undef X
  }
  return -1;
}

static const MvmOpInfo* mvm_op(int id) {  // nullptr if id is not registered
  for (const MvmOpInfo& o : kMvmOps) if (o.id == id) return &o;
  return nullptr;
}

// ---------------- errors ----------------
#define MVM_ERRORS(E) \
  E(OK, "ok") E(BAD_OP, "op_id not in registry") E(BAD_ARITY, "args_words != operator arity") \
  E(BAD_BUF_INDEX, "buffer index outside the device table") E(BAD_BUF, "buffer not f32 / ld != cols") \
  E(BOUNDS, "element count exceeds buffer") E(BAD_VAR, "version variable out of range") \
  E(VERSION_GAP, "pub_ver != version[pub_var]+1 (publisher not contiguous or two writers)") \
  E(WAIT_VIOLATION, "task ran with wait_var/wait_ver unsatisfied") E(DOUBLE_EXEC, "task executed twice") \
  E(BAD_RECORD, "malformed admission record") E(BAD_TASK, "task descriptor out of range (succ/args/priority/indegree)") \
  E(SLOTS, "task slot pool exhausted (max_tasks)") E(POOL, "successor/args pool exhausted") \
  E(BATCH_TOO_LARGE, "bundle larger than live_cap") E(TIMEOUT, "device timeout (deadlock or stalled admission)") \
  E(QUEUE_OVERFLOW, "ready ring invariant violated") E(BAD_CONFIG, "bad launch configuration") \
  E(PARK_STALL, "more tasks parked on versions than the park budget (live_cap) and admission needs room") E(TASK_LIMIT, "a task ran longer than task_limit") \
  E(ALIAS_RECORD, "a record other than the prestaged one under an aliased successor ring (succ_alias)")
enum MvmErr {
#define E(n, s) MVM_ERR_##n,
  MVM_ERRORS(E)
#undef E
};

// ---------------- shared host/device state ----------------
struct MvmTrace {  // one per slot, 64 B
  unsigned long long admit_ns, ready_ns, start_ns, end_ns;  // %globaltimer
  int smid, block, ver0, ver1;                              // ver0/ver1: wait_var version at start / end (-1 none)
  int flags, exec_count, pad0, pad1;                        // flags bit0 = deadline missed
#ifdef MVM_PROF_GAP
  int gp[18];   // profiling build only: gp[0..8] SM-clock cycles of the phases between the previous task on this block and this one (see MVM_PG); gp[9..14] sub-steps (MVM_PG_SUB); gp[15] valid; gp[16] the SM cycle counter beside start_ns
#endif
};
#define MVM_TRACE_MISSED 1
#define MVM_TRACE_CONT   16   // started as the continuation of the task that released it (no enqueue, no pop)

#define MVM_POP_EXIT (-2)   // mvm_pop / mvm_fifo_pop: this block should leave its loop
#define MVM_FIFO_SHIFT 26   // FIFO mode (MvmDevState::fifo_only): a ready-ring cell's seq word holds (pos + 1) << MVM_FIFO_SHIFT | slot, published in one store
struct MvmCell { unsigned long long seq, ready_ns, deadline_ns; int slot, res; };   // 32 B; deadline_ns 0 = none, res = MvmRes (SCHED)
struct alignas(128) MvmQCtl { unsigned long long head, tail, avail; };             // own cache line; avail = published minus reserved tickets (mvm_claim)

struct MvmWin { int remaining; unsigned succ_end; int ne; int pad; };                              // per admitted bundle record ("window")

struct MvmCounters {  // device memory; hot fields on separate lines
  alignas(128) int done;
  alignas(128) int admitted, admit_final;
  alignas(128) int err_code, err_slot, err_a, err_b;
  alignas(128) int parked;                                                        // tasks parked on a version (not counted against live_cap)
  alignas(128) unsigned long long ftail;                                          // free-slot ring tail (many producers)
};
struct MvmWatch { unsigned long long since, checked; int done, admitted; };   // per-block progress watch (thread 0)
struct MvmHostErr { volatile int slot, a, b; volatile int code; };  // mapped mirror of the first device error

struct MvmDistDev;  // MVM_DIST_HOOK: include/mvm_dist.cuh
#define MVM_ARG_STRIDE 20   // argument words reserved per slot (>= MVM_MAX_ARGS)
static_assert(MVM_ARG_STRIDE >= MVM_MAX_ARGS, "slot argument stride");

// nslots, succ_cap, win_cap and live_cap are powers of two.
struct MvmDevState {
  int nslots, live_cap, nbufs, ring_words, num_vars, win_cap, rcap, pad1;   // rcap = ready-ring depth = 4 * live_cap (see MvmCounters::parked)
  unsigned succ_cap;
  long long aging_ns;
  int cont_on;     // 1: a worker runs one released successor directly (continuation), when the ready queue is empty
  int fifo_only;   // 1: every task is priority 0 with no deadline and the scheduler is off: one FIFO ring, each idle worker waits on the cell of its own ticket
  int no_recycle;  // 1: one record holds the whole run and slots are not reused: a finished task only counts done (no window credit, no free-ring push)
  int no_waits;    // 1: no task of the run waits on a version: releasing a task needs no look at its descriptor
  int trace_on;    // 1: per-task timeline and witness in trace[] (needed by --trace-csv, tests and the scheduler); 0: a run that only times keeps none of it off the task path
  int prestaged;   // 1: the host wrote the (single, one-record) run's record into device memory before the kernel started: the admitter skips its staging pass
  int succ_alias;  // 1: that prestaged record's successor words are the successor ring (succ points at them in stage; the host validated each one): the admitter
                  //    skips its copy, which is an identity there (slot = record index, ring offset 0) and ran on one block alone (gpt-162m: 8.5M words, 5 ms)
  int warm_only;   // 1: a launch that returns at once: the kernel's code is uploaded and started before the timed launch
#ifdef MVM_PROF_GAP
  int noop_ops;    // profiling only (--noop-ops): the graph, the dependencies and the dispatcher run as usual but no op body does, to separate the hop's own cost from what the op bodies leave behind
#endif
  unsigned long long timeout_ns, task_limit_ns;   // stall timeout; per-task running-time limit (0 = off, checked when the task ends)
  const MvmBuf* bufs;
  MvmTaskDesc* tasks;                               // [nslots]
  int *indeg, *state, *park_next, *args, *adm_slots;  // [nslots], args [nslots*MVM_ARG_STRIDE], adm_slots [live_cap]
  int *succ;                                        // successor ring [succ_cap] of slot ids
  int *version, *park_head;                         // [num_vars]
  MvmTrace* trace;                                  // [nslots]: the latest occupant of each slot
  MvmCell *cells, *fcells;                          // ready rings [nc*MVM_NBUCKETS*rcap], free-slot ring [nslots]
  MvmQCtl* qctl;
  MvmWin* wins;                                     // [win_cap]
  MvmCounters* cnt;
  // host-visible (mapped pinned) control block
  int* ring;
  unsigned long long *doorbell, *ring_head;
  unsigned int *closed, *win_done;                  // win_done[w & (win_cap-1)] == w+1 once window w has finished
  MvmHostErr* herr;
  MvmSchedPtrs sp;   // SCHED: scheduler state; sp.on == 0 reproduces the original single-queue executor
  const MvmDistDev* dist;  // MVM_DIST_HOOK: nullptr = single-device executor (all dist code is skipped)
  int* stage;              // [ring_words] device copy of the record being admitted (the admitter bulk-copies it from the mapped ring first)
};

#ifdef __CUDACC__
// ---------------- PTX memory primitives ----------------
__device__ __forceinline__ unsigned long long mvm_now() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int mvm_smid() { unsigned r; asm("mov.u32 %0, %%smid;" : "=r"(r)); return (int)r; }
__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.b32 %0,[%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned long long ld_acq64(const unsigned long long* p) { unsigned long long v; asm volatile("ld.acquire.gpu.global.b64 %0,[%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(int* p, int v) { asm volatile("st.release.gpu.global.b32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rel64(unsigned long long* p, unsigned long long v) { asm volatile("st.release.gpu.global.b64 [%0],%1;" ::"l"(p), "l"(v) : "memory"); }
__device__ __forceinline__ void red_add64(unsigned long long* p, unsigned long long v) { asm volatile("red.relaxed.gpu.global.add.u64 [%0],%1;" ::"l"(p), "l"(v) : "memory"); }   // posted: the issuing thread does not wait for the L2 round trip
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
__device__ __noinline__ void mvm_fail(MvmCounters* cnt, MvmHostErr* herr, int code, int slot, int a, int b) {
  if (atomicCAS(&cnt->err_code, 0, code) != 0) return;
  cnt->err_slot = slot; cnt->err_a = a; cnt->err_b = b;
  herr->slot = slot; herr->a = a; herr->b = b;
  st_rel_sys(&herr->code, code);
}
__device__ __forceinline__ void mvm_fail(const MvmDevState& S, int code, int slot, int a = 0, int b = 0) { mvm_fail(S.cnt, S.herr, code, slot, a, b); }

// ---------------- operator context and typed arguments ----------------
struct MvmView { float* p; int rows, cols; };    // f32 buffer; p == nullptr: absent (buf?) or invalid (already reported)
struct MvmIView { int* p; int rows, cols; };     // i32 buffer

struct MvmCtx {
  const MvmBuf* bufs;
  int nbufs, slot;
  MvmCounters* cnt;
  MvmHostErr* herr;
  float* smem;  // MVM_SMEM_FLOATS floats, block-shared scratch
  __device__ void fail(int code, int a = 0, int b = 0) const { mvm_fail(cnt, herr, code, slot, a, b); }
  // Typed accessors. OP/I are checked against the registry at compile time (see MVM_OP): a device function that reads an
  // argument as the wrong type, or past the registered arity, does not compile. buf = required, buf_opt = a `buf?` argument.
  // dtype must match (f32 views for buf/buf_opt, i32 views for ibuf/ibuf_opt); a mismatch is MVM_ERR_BAD_BUF.
  template <int OP, int I> __device__ MvmView buf(const int32_t* a) const {
    static_assert(I < mvm_op_arity(OP) && mvm_op_argtype(OP, I) == MVM_A_BUF, "argument is not a required buf in the registry");
    MvmIView v = resolve(a[I], 0, false, I); return {(float*)v.p, v.rows, v.cols};
  }
  template <int OP, int I> __device__ MvmView buf_opt(const int32_t* a) const {
    static_assert(I < mvm_op_arity(OP) && mvm_op_argtype(OP, I) == MVM_A_BUF_OPT, "argument is not a `buf?` in the registry");
    MvmIView v = resolve(a[I], 0, true, I); return {(float*)v.p, v.rows, v.cols};
  }
  template <int OP, int I> __device__ MvmIView ibuf(const int32_t* a) const {
    static_assert(I < mvm_op_arity(OP) && mvm_op_argtype(OP, I) == MVM_A_BUF, "argument is not a required buf in the registry");
    return resolve(a[I], 1, false, I);
  }
  template <int OP, int I> __device__ MvmIView ibuf_opt(const int32_t* a) const {
    static_assert(I < mvm_op_arity(OP) && mvm_op_argtype(OP, I) == MVM_A_BUF_OPT, "argument is not a `buf?` in the registry");
    return resolve(a[I], 1, true, I);
  }
  template <int OP, int I> __device__ int i32(const int32_t* a) const {
    static_assert(I < mvm_op_arity(OP) && mvm_op_argtype(OP, I) == MVM_A_I32, "argument is not an i32 in the registry");
    return a[I];
  }
  template <int OP, int I> __device__ float f32(const int32_t* a) const {
    static_assert(I < mvm_op_arity(OP) && mvm_op_argtype(OP, I) == MVM_A_F32, "argument is not an f32 in the registry");
    return __int_as_float(a[I]);
  }
  // 0 <= n <= elements of every view; false (after reporting) otherwise. An invalid view was already reported by buf().
  template <class... V> __device__ bool fits(int n, V... v) const {
    if ((... || (v.p == nullptr))) return false;
    bool ok = n >= 0;
    ((ok = ok && (long long)n <= (long long)v.rows * v.cols), ...);
    if (!ok) fail(MVM_ERR_BOUNDS, n);
    return ok;
  }
 private:
  __device__ MvmIView resolve(int idx, int dtype, bool opt, int arg) const {
    if (idx == -1 && opt) return {nullptr, 0, 0};                                   // absent
    if (idx < 0) { fail(MVM_ERR_BAD_BUF, idx, arg); return {nullptr, 0, 0}; }       // required buffer given as "absent"
    if (idx >= nbufs) { fail(MVM_ERR_BAD_BUF_INDEX, idx, arg); return {nullptr, 0, 0}; }
    const MvmBuf& b = bufs[idx];
    if (b.dtype != dtype || b.ld != b.cols || b.p == nullptr) { fail(MVM_ERR_BAD_BUF, idx, arg); return {nullptr, 0, 0}; }
    return {(int*)b.p, b.rows, b.cols};
  }
};

// The kernel an op body is compiled into: the dynamic executor's (mvm_exec_kernel) or the static schedule's (mvm_static_kernel). In these
// whole-program kernels every form an op compiles grows the kernel that dispatches it, so an op may compile a leaner form into one of them
// (matmul_tile leaves its 256x128 fast forms out of the static kernel, whose rows never take that tile).
enum MvmKernelKind : int { MVM_IN_EXEC = 0, MVM_IN_STATIC = 1 };
#define MVM_OP(n) template <int MVM_SELF = MVM_OP_##n, int MVM_KERNEL = MVM_IN_EXEC> __device__ __forceinline__ void op_##n(const MvmCtx& c, const int32_t* a)
#define ARG_BUF(i) c.template buf<MVM_SELF, (i)>(a)
#define ARG_BUF_OPT(i) c.template buf_opt<MVM_SELF, (i)>(a)   /* .p == nullptr: absent */
#define ARG_IBUF(i) c.template ibuf<MVM_SELF, (i)>(a)
#define ARG_IBUF_OPT(i) c.template ibuf_opt<MVM_SELF, (i)>(a)
#define ARG_I32(i) c.template i32<MVM_SELF, (i)>(a)
#define ARG_F32(i) c.template f32<MVM_SELF, (i)>(a)

// ---------------- operators (block-cooperative: every thread of the block calls with the same args) ----------------
MVM_OP(noop) {}
MVM_OP(copy) {
  MvmView s = ARG_BUF(0), d = ARG_BUF(1); int n = ARG_I32(2);
  if (!c.fits(n, s, d)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) d.p[i] = s.p[i];
}
MVM_OP(add) {
  MvmView x = ARG_BUF(0), y = ARG_BUF(1), o = ARG_BUF(2); int n = ARG_I32(3);
  if (!c.fits(n, x, y, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = x.p[i] + y.p[i];
}
MVM_OP(scale) {
  MvmView x = ARG_BUF(0), o = ARG_BUF(1); float al = ARG_F32(2); int n = ARG_I32(3);
  if (!c.fits(n, x, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = al * x.p[i];
}
MVM_OP(relu) {
  MvmView x = ARG_BUF(0), o = ARG_BUF(1); int n = ARG_I32(2);
  if (!c.fits(n, x, o)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = x.p[i] > 0.f ? x.p[i] : 0.f;
}
MVM_OP(sgd) {
  MvmView w = ARG_BUF(0), g = ARG_BUF(1); float lr = ARG_F32(2); int n = ARG_I32(3);
  if (!c.fits(n, w, g)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) w.p[i] = w.p[i] - lr * g.p[i];
}
MVM_OP(predict) {
  MvmView x = ARG_BUF(0), w = ARG_BUF(1), o = ARG_BUF(2); int n = ARG_I32(3);
  if (!c.fits(n, x, o) || !c.fits(1, w)) return;
  float w0 = w.p[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x) o.p[i] = w0 * x.p[i];
}
// Deterministic reduction order (the CPU reference in tests replicates it): thread t sums elements t, t+256, ... in order;
// then a halving tree over the 256 partials.
MVM_OP(mse_grad) {
  MvmView p = ARG_BUF(0), t = ARG_BUF(1), x = ARG_BUF(2), loss = ARG_BUF(3), grad = ARG_BUF(4); int n = ARG_I32(5);
  if (n < 1) { c.fail(MVM_ERR_BOUNDS, n); return; }
  if (!c.fits(n, p, t, x) || !c.fits(1, loss, grad)) return;
  float* sl = c.smem;
  float* sg = c.smem + MVM_THREADS;
  float la = 0.f, ga = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float d = p.p[i] - t.p[i];
    la += d * d;
    ga += (2.f * d) * x.p[i];
  }
  sl[threadIdx.x] = la; sg[threadIdx.x] = ga;
  __syncthreads();
  for (int s = MVM_THREADS / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) { sl[threadIdx.x] += sl[threadIdx.x + s]; sg[threadIdx.x] += sg[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { loss.p[0] = sl[0] / (float)n; grad.p[0] = sg[0] / (float)n; }
}
// fp32, 16x16 smem tiles, any m,n,k. Each output element accumulates k ascending from 0 (same order as a naive loop).
MVM_OP(matmul) {
  MvmView A = ARG_BUF(0), B = ARG_BUF(1), O = ARG_BUF(2); int m = ARG_I32(3), n = ARG_I32(4), k = ARG_I32(5);
  if (A.p == nullptr || B.p == nullptr || O.p == nullptr) return;
  if (m < 1 || n < 1 || k < 1 || m > A.rows || k > A.cols || k > B.rows || n > B.cols || m > O.rows || n > O.cols) {
    c.fail(MVM_ERR_BOUNDS, m, n); return;
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
MVM_OP(fill) {
  MvmView x = ARG_BUF(0); float v = ARG_F32(1); int n = ARG_I32(2);
  if (!c.fits(n, x)) return;
  for (int i = threadIdx.x; i < n; i += blockDim.x) x.p[i] = v;
}
MVM_OP(spin) {
  int us = ARG_I32(0);
  if (us < 0 || us > 10000000) { c.fail(MVM_ERR_BOUNDS, us); return; }   // at most 10 s: a task cannot be interrupted
  if (threadIdx.x == 0) { unsigned long long end = mvm_now() + (unsigned long long)(us < 0 ? 0 : us) * 1000ull; while (mvm_now() < end) {} }
}

// Optional operator packs: each defines MVM_OP(name) for every X line of its .def.
#include "mvm_dist.cuh"  // MVM_DIST_HOOK: phase 1 (types, primitives); phase 2 (inbox pump) follows mvm_publish
#if __has_include("mvm_ops_comm.cuh") && !defined(MVM_LEAN)
#include "mvm_ops_comm.cuh"  // MVM_DIST_HOOK
#endif
#if __has_include("mvm_ops_gpt.cuh")
#include "mvm_ops_gpt.cuh"
#endif
#if __has_include("mvm_ops_extra.cuh") && !defined(MVM_LEAN)      // MVM_MODELS_HOOK: extra generic pack (needs the gpt pack's helpers and dynamic smem)
#include "mvm_ops_extra.cuh"
#endif
#if defined(MVM_WITH_GAME) && __has_include("mvm_ops_game.cuh")
#include "mvm_ops_game.cuh"
#endif
#ifdef MVM_EXTRA_OPS_CUH
#include MVM_EXTRA_OPS_CUH
#endif

// The fused MLP ops run only through the static-schedule path (mvm_static.cuh calls mvm_fused_mlp.cuh directly, with the run's descriptor): the generic dispatch refuses them.
#define X(n, id, sig) MVM_OP(n) { (void)a; c.fail(MVM_ERR_BAD_OP, MVM_SELF); }
#include "mvm_ops_fmlp.def"
#undef X

// Dynamic shared memory every launched block must provide: the maximum any operator pack asks for (static_cast: the packs
// define these as constant expressions). Host code passes it at launch and raises the per-function limit when above 48 KiB.
#if defined(MVM_GPT_SMEM_BYTES) && defined(MVM_EXTRA_SMEM_BYTES)
#define MVM_DYN_SMEM_BYTES ((size_t)(MVM_GPT_SMEM_BYTES) > (size_t)(MVM_EXTRA_SMEM_BYTES) ? (size_t)(MVM_GPT_SMEM_BYTES) : (size_t)(MVM_EXTRA_SMEM_BYTES))
#elif defined(MVM_GPT_SMEM_BYTES)
#define MVM_DYN_SMEM_BYTES ((size_t)(MVM_GPT_SMEM_BYTES))
#elif defined(MVM_EXTRA_SMEM_BYTES)
#define MVM_DYN_SMEM_BYTES ((size_t)(MVM_EXTRA_SMEM_BYTES))
#else
#define MVM_DYN_SMEM_BYTES ((size_t)0)
#endif

// Adding an operator = one X line in mvm_ops.def + one MVM_OP(name) above; a missing MVM_OP is a compile error here.
// Not inlined: with every op body inlined into the worker loop, the loop's own state (the run's flags and pointers, the task it owes, its ticket) was spilled to the stack
// around every op and reloaded one dependent load at a time on every hop. As its own function the dispatch gets the op bodies' register allocation and the loop keeps its own.
// -DMVM_OPS_ONLY='"noop,matmul_tile_bf16,..."' builds only the listed op bodies into the dispatch (the others fail with MVM_ERR_BAD_OP, and the
// CLI refuses a bundle that uses one): every op body is inlined into this one function, so a full build hands ptxas the whole registry at once
// (Nov 15: sm_90 ptxas ran past 20 minutes at 56 GB). Unset, every op is built.
__host__ __device__ constexpr bool mvm_op_built(const char* n) {
#ifdef MVM_OPS_ONLY
  for (const char* l = MVM_OPS_ONLY; *l;) {
    const char* p = n; const char* q = l;
    while (*p && *q && *q != ',' && *p == *q) { ++p; ++q; }
    if (!*p && (!*q || *q == ',')) return true;
    while (*l && *l != ',') ++l;
    if (*l == ',') ++l;
  }
  return false;
#else
  (void)n; return true;
#endif
}
// KERNEL: the kernel calling it (MvmKernelKind), passed to every op as MVM_KERNEL.
template <int KERNEL>
__device__ __noinline__ void mvm_dispatch(int op, const MvmCtx c, const int32_t* a) {
  switch (op) {
#define X(n, id, sig) case id: if constexpr (mvm_op_built(#n)) op_##n<MVM_OP_##n, KERNEL>(c, a); else c.fail(MVM_ERR_BAD_OP, op); break;
#include "mvm_ops_list.inc"
#undef X
    default: c.fail(MVM_ERR_BAD_OP, op);
  }
}

// Device-side registry lookups (admission validation): -1 / 0 for an unregistered id.
__device__ __forceinline__ int mvm_dev_arity(int op) { return mvm_op_arity(op); }
__device__ __forceinline__ uint64_t mvm_dev_types(int op) {
  switch (op) {
#define X(n, id, sig) case id: return mvm_sig::types(sig);
#include "mvm_ops_list.inc"
#undef X
  }
  return 0;
}

// ---------------- ready queue and scheduling policy ----------------
// SCHED: mvm_enqueue / mvm_pop (per-class rings, tiers, caps) and the policy block live in mvm_sched.cuh.
#include "mvm_sched.cuh"

// ---------------- free-slot ring (slot recycling) ----------------
// A bounded MPSC ring of free slot ids: workers push when a task completes (mvm_post_task), the admitter (block 0) pops. Cell for position
// pos holds a slot when seq == pos+1; the consumer resets it to pos+nslots, which is what the producer of pos+nslots waits for.
// At most nslots ids exist, so a producer never finds its cell occupied for longer than the consumer's pop takes.

// ---------------- release: indegree reached 0 ----------------
// Enqueue now if the version wait is satisfied, else park the task on its variable's list.
//
// Parked tasks live on a lock-free push / detach-all list per variable. EVERY node on a list is owned by whoever detaches
// it, and a node is removed from the list only by a detach, so a slot is never reachable from a list after its task was
// enqueued (this is what makes slot recycling safe; an earlier design left claimed nodes on the list).
// Lost wakeup: waiter = push; fence.sc; load version.   publisher = store version; fence.sc; detach.
// By the fence order at least one side sees the other: either the waiter sees the new version and scans the list itself, or
// the publisher's detach sees the node. A scan re-parks nodes whose version is still too low and re-checks them the same way.
__device__ __forceinline__ void mvm_scan_parked(const MvmDevState& S, int var) {
  bool again;
  do {
    again = false;
    int n = atom_exch(S.park_head + var, -1);
    while (n >= 0) {
      int next = S.park_next[n];                                 // read before the node is re-pushed or its task enqueued
      int need = S.tasks[n].wait_ver;
      if (ld_acq(S.version + var) >= need) {
        if (atom_cas(S.state + n, 1, 2) != 1) mvm_fail(S, MVM_ERR_DOUBLE_EXEC, n, 1, 0);
        else { atom_add(&S.cnt->parked, -1); mvm_sched_unpark(S, n); mvm_enqueue(S, n); }
      } else {
        int* head = S.park_head + var; int old;
        do { old = *(volatile int*)head; S.park_next[n] = old; } while (atom_cas(head, old, n) != old);
        __threadfence();
        if (ld_acq(S.version + var) >= need) again = true;       // became satisfied meanwhile: detach again
      }
      n = next;
    }
  } while (again);
}
__device__ __forceinline__ void mvm_release_task(const MvmDevState& S, int slot, int var, int need) {   // var, need: the task's wait_var and wait_ver (the caller read them with the rest of its descriptor)
  if (var < 0 || ld_acq(S.version + var) >= need) { mvm_enqueue(S, slot); return; }
  S.state[slot] = 1;
  atom_add(&S.cnt->parked, 1);
  mvm_sched_park(S, slot);   // SCHED: count the parked waiter (priority inheritance input) before it is reachable from the list
  int* head = S.park_head + var; int old;
  do { old = *(volatile int*)head; S.park_next[slot] = old; } while (atom_cas(head, old, slot) != old);
#ifndef MVM_BREAK_PARK_RECHECK  // negative control for the litmus test: a waiter that never re-checks loses wakeups
  __threadfence();  // seq_cst fence: push -> fence -> load version
  if (ld_acq(S.version + var) >= need) mvm_scan_parked(S, var);
#endif
}
__device__ __forceinline__ void mvm_release_task(const MvmDevState& S, int slot) { const MvmTaskDesc* d = S.tasks + slot; mvm_release_task(S, slot, d->wait_var, d->wait_ver); }

// version[pub_var]: N-1 -> N. Caller has passed a block barrier after the op's stores.
__device__ __forceinline__ void mvm_publish(const MvmDevState& S, int slot, int var, int ver) {
  __threadfence();                                              // fence.sc.gpu: weights before version (see header)
  int old = atom_cas(S.version + var, ver - 1, ver);            // release store, and detects gaps / double writers
  if (old != ver - 1) { mvm_fail(S, MVM_ERR_VERSION_GAP, slot, old, ver); return; }
  __threadfence();                                              // version store -> fence -> list detach
  mvm_scan_parked(S, var);
}

// ---------------- worker ----------------
struct MvmShared {
  MvmTaskDesc d;
  int cmd, skip, how, part, cont, qbusy, err;   // how: MVM_TRACE_* start reason from mvm_pop (| MVM_HOW_BW); part: STATIC-mode class of this worker's SM; cont: slot this block runs next without a pop; qbusy: the ready queue had work when this task started; err: the error code as of the last dispatch (read overlapped with the descriptor)
  unsigned long long start_ns;   // this task's start time: the task-limit check needs it after the op (from a register, not the trace in global memory)
#ifdef MVM_PROF_GAP
  long long pg[16]; int pg_valid;
  int sub[MVM_PG_SUBS];   // the slowest lane's completion of each sub-step of a phase, in SM cycles after the phase began (MVM_PG_SUB)
#endif
  float smem[MVM_SMEM_FLOATS];
};
// MVM_PROF_GAP: thread 0 stamps clock64() at the phase boundaries of the dispatch gap (previous op end -> this op start) and stores
// the deltas in this task's trace. 0 op end | 1 end bookkeeping+publish | 2 successors released | 3 sched_done | 4 window credit |
// 5 slot freed | 6 done counted | 7 pop returned | 8 descriptor copied | 9 start bookkeeping.
#ifdef MVM_PROF_GAP
#define MVM_PG(k) do { if (threadIdx.x == 0) sh.pg[k] = clock64(); } while (0)
#define MVM_PG_SUB(i, base) atomicMax(&sh.sub[i], (int)(clock64() - sh.pg[base]))   // this lane finished sub-step i of the phase that began at marker `base`
#else
#define MVM_PG(k) do { } while (0)
#define MVM_PG_SUB(i, base) do { } while (0)
#endif

// Stall detector (one per thread 0 of a block): `timeout` is the longest the executor may go without a single task finishing or
// being admitted while work is pending. Waiting idle for the host is never a stall, and a long-lived service is never killed for
// its age. `timeout` must therefore exceed the longest single task (a task cannot be interrupted; see task_limit).
// require_pending=false (the admitter waiting for room) also counts "nothing pending but still no room" as a stall.
__device__ __forceinline__ bool mvm_stalled(const MvmDevState& S, MvmWatch& w, bool require_pending = true) {
  const unsigned long long n = mvm_now();
  if (n - w.checked < 1000000ull) return false;   // look at the hot counters at most once per ms: idle workers poll this in a tight loop
  w.checked = n;
  const int d = ld_acq(&S.cnt->done), a = ld_acq(&S.cnt->admitted);
  if (d != w.done || a != w.admitted || (require_pending && a == d)) { w.done = d; w.admitted = a; w.since = n; return false; }
  return n - w.since > S.timeout_ns;
}
__device__ __forceinline__ bool mvm_should_exit(const MvmDevState& S, MvmWatch& t0) {  // (declared in mvm_sched.cuh)
  if (*(volatile int*)&S.cnt->err_code) return true;
  if (mvm_stalled(S, t0)) {
    mvm_fail(S, ld_acq(&S.cnt->parked) > S.live_cap ? MVM_ERR_PARK_STALL : MVM_ERR_TIMEOUT, -1, ld_acq(&S.cnt->admitted), ld_acq(&S.cnt->done));
    return true;
  }
  if (ld_acq(&S.cnt->admit_final)) return ld_acq(&S.cnt->done) >= ld_acq(&S.cnt->admitted);
  return false;
}

#define MVM_DIST_PUMP_PHASE 1  // MVM_DIST_HOOK: remote-publish pump, after mvm_stalled and mvm_scan_parked (mvm_dist_poll / mvm_dist_pump)
#include "mvm_dist.cuh"

__device__ void mvm_run_task(const MvmDevState& S, int slot, MvmShared& sh) {
  MVM_PG(7); MVM_PG(8);   // sh.d was filled from the pop's prefetched words and published by the barrier that follows the pop
  const MvmTaskDesc& d = sh.d;
  MvmTrace* tr = S.trace + slot;
  if (threadIdx.x == 0) {
    bool bad = false;
    if (S.trace_on) {
      sh.start_ns = mvm_now(); tr->start_ns = sh.start_ns; tr->smid = mvm_smid(); tr->block = blockIdx.x; tr->ver0 = tr->ver1 = -1;
      tr->flags = sh.how & ~MVM_HOW_BW;                                 // a plain store: admission zeroed the trace, and a read-modify-write here is an L2 round trip before the op
      red_add(&tr->exec_count, 1);   // witness: a posted add; mvm_post_task checks it is exactly 1 (a second run of the slot is reported there)
    }
    mvm_sched_start(S, slot, d.workload_id, sh.how, sh.start_ns);
    if (d.wait_var >= 0) {
      int v = ld_acq(S.version + d.wait_var);  // acquire: pairs with the publisher's release (header comment)
      if (S.trace_on) tr->ver0 = v;
      if (v < d.wait_ver) { mvm_fail(S, MVM_ERR_WAIT_VIOLATION, slot, v, d.wait_ver); bad = true; }
    }
#ifdef MVM_BREAK_FENCE  // negative test: publish BEFORE the op's stores (no happens-before) so version tests must fail
    if (d.pub_var >= 0) mvm_publish(S, slot, d.pub_var, d.pub_ver);
#endif
    sh.skip = bad || sh.err;      // sh.err was read during the pop (or the continuation's descriptor load), so no load sits on the path to the op
    sh.cont = -1;
  }
  __syncthreads();
  MVM_PG(9);
  // Prefetch this thread's first successor id: the list is immutable once admitted, so the round trip hides behind the op instead of sitting
  // on the dependent-hop critical path. It is only consumed after the op.
  int sid0 = -1;
  if ((int)threadIdx.x < d.succ_cnt) sid0 = S.succ[(unsigned)(d.succ_off + threadIdx.x) & (S.succ_cap - 1)];
  // Continuation only while the ready queue is empty (looked at before the op, one coalesced load by warp 0): a chain must not starve queued
  // tasks of other priorities. Idle workers take anything that arrives during the op.
  if (threadIdx.x < 32) {
    const unsigned b = __ballot_sync(0xffffffffu, S.cont_on && !S.sp.on && !S.fifo_only && (long long)*(volatile unsigned long long*)&S.qctl[threadIdx.x].avail > 0);
    if (threadIdx.x == 0) sh.qbusy = b != 0;
  }
#ifdef MVM_PROF_GAP
  if (threadIdx.x == 0) {   // gp[k] = phase k -> k+1 (markers 0..6 were stamped by the previous task of this block)
    for (int k = 0; k < 18; ++k) tr->gp[k] = 0;
    if (sh.pg_valid) { for (int k = 0; k < 9; ++k) tr->gp[k] = (int)(sh.pg[k + 1] - sh.pg[k]); for (int k = 0; k < MVM_PG_SUBS; ++k) tr->gp[9 + k] = sh.sub[k]; tr->gp[15] = 1; }
    for (int k = 0; k < MVM_PG_SUBS; ++k) sh.sub[k] = 0;
    tr->gp[16] = (int)sh.pg[9];   // the SM's cycle counter beside this task's start_ns: two tasks of one block give the clock the phases are counted at
    sh.pg_valid = 1;
  }
#endif
  if (!sh.skip) {
    MvmCtx c{S.bufs, S.nbufs, slot, S.cnt, S.herr, sh.smem};
#ifdef MVM_PROF_GAP
    if (!S.noop_ops)
#endif
    mvm_dispatch<MVM_IN_EXEC>(d.op_id, c, S.args + d.args_off);
  }
  // Every warp makes ITS OWN op stores visible GPU-wide before the barrier: bar.sync orders them only within the block, and the single
  // fence of the publishing / releasing thread orders just that thread's stores, so without this a successor or a version reader on
  // another SM could still see stale data from the other warps (the `versions` suite's rare "reader violated its snapshot" failure,
  // present in the original executor at ~1 in 40 runs and made likelier by a shorter dispatch path). A task nobody waits on skips it.
#ifndef MVM_BREAK_FENCE
  if (!sh.skip && (d.succ_cnt > 0 || d.pub_var >= 0)) __threadfence();
#endif
  __syncthreads();  // all op stores of the block are visible before thread 0 publishes / the successors are released
  MVM_PG(0);
  if (threadIdx.x == 0) {
    unsigned long long end = 0;
    if (S.trace_on) {
      end = mvm_now();
      tr->end_ns = end;
      if (d.wait_var >= 0) tr->ver1 = ld_acq(S.version + d.wait_var);
      const bool missed = d.deadline_us >= 0 && end - tr->admit_ns > (unsigned long long)d.deadline_us * 1000ull;
      if (missed) tr->flags |= MVM_TRACE_MISSED;
    }
    if (S.trace_on && S.task_limit_ns && end - sh.start_ns > S.task_limit_ns) mvm_fail(S, MVM_ERR_TASK_LIMIT, slot, (int)((end - sh.start_ns) / 1000), (int)(S.task_limit_ns / 1000));
#ifndef MVM_BREAK_FENCE
    if (d.pub_var >= 0 && !sh.skip) mvm_publish(S, slot, d.pub_var, d.pub_ver);  // publish THEN wake successors
#endif
  }
  __syncthreads();
  MVM_PG(1);
  if (!sh.skip && (int)threadIdx.x < d.succ_cnt) {
    for (int e = threadIdx.x; e < d.succ_cnt; e += blockDim.x) {
      int s = e < blockDim.x ? sid0 : S.succ[(unsigned)(d.succ_off + e) & (S.succ_cap - 1)];
      MVM_PG_SUB(0, 1);   // the successor id is in hand
      const int left = atom_add(S.indeg + s, -1);
      MVM_PG_SUB(1, 1);   // the indegree atomic returned
      if (left == 1) {
        // Continuation: one released successor of the same priority bucket whose version wait is met is run by this block directly: no enqueue,
        // no pop. Only without the scheduler (it accounts ready/running per class at enqueue and pop), and only when the queue was empty at the
        // start of this task (a chain must not starve queued tasks).
        const MvmTaskDesc* sd = S.tasks + s;
        int s_pri = 0, s_wv = -1, s_wn = 0;
        if (!(S.fifo_only && S.no_waits)) { s_pri = sd->priority; s_wv = sd->wait_var; s_wn = sd->wait_ver; }   // read together, one round trip; a run with no version waits and no priorities needs no look at the successor
#ifndef MVM_BREAK_CONT
        const bool s_ver_met = s_wv < 0 || ld_acq(S.version + s_wv) >= s_wn;
#else   // negative control (tests/litmus/run_litmus.sh): the continuation ignores the successor's version wait (e2e versions must report it)
        const bool s_ver_met = true;
#endif
        if (S.cont_on && !S.sp.on && !sh.qbusy && s_pri == d.priority && s_ver_met &&
            atomicCAS(&sh.cont, -1, s) == -1) {
          if (S.trace_on) S.trace[s].ready_ns = mvm_now();
        } else { mvm_sched_pref(S, s); mvm_release_task(S, s, s_wv, s_wn); MVM_PG_SUB(2, 1); }   // SCHED: affinity = this block
      }
    }
  }
  __syncthreads();
  MVM_PG(2);
  // The class's running/BW counters feed the very next pop of this block, so they are released here, ahead of it. Everything else the
  // task still owes (statistics, window credit, slot recycling, done count) is mvm_post_task, run by a helper thread during the pop.
  if (threadIdx.x == 0) {
    mvm_sched_release(S, d.workload_id, sh.how & MVM_HOW_BW);
    MVM_PG(3); MVM_PG(4); MVM_PG(5); MVM_PG(6);
  }
}

// Completion bookkeeping of task `slot` (one thread, overlapped with the next pop or the continuation's descriptor load; the slot is not recycled before the
// free-ring store below, so its trace and descriptor are intact). Everything is issued first, in one go, so the round trips overlap: the window credit (`wid` comes
// from a register, so no descriptor load precedes it), the free-ring ticket and the witness load. Their results are consumed after all of them are in flight. `done`
// is a posted add: nobody reads an old value, and a worker exits only after its own bookkeeping ran (the barrier after the pop).
__device__ void mvm_post_task(const MvmDevState& S, int slot, unsigned wid) {
  const MvmTrace* tr = S.trace + slot;
  if (S.sp.on) mvm_sched_account(S, slot, tr->start_ns, tr->end_ns, tr->flags & MVM_TRACE_MISSED);   // SCHED: statistics and estimates (before the slot is recycled)
  if (S.no_recycle) { red_add(&S.cnt->done, 1); return; }                                // the whole run is one record: no window to return, no slot to recycle (nothing reads the old done value)
  const int witness = S.trace_on ? mvm_vol(&tr->exec_count) : 1;
  const int window_left = atom_add(&S.wins[wid & (S.win_cap - 1)].remaining, -1);       // returns the window credit, which also lets the admitter reclaim the window's successor storage
  const unsigned long long pos = atom_add64(&S.cnt->ftail, 1ull);                       // the slot's place in the free ring
  if (witness != 1) mvm_fail(S, MVM_ERR_DOUBLE_EXEC, slot, witness);
  if (window_left == 1) st_rel_sys((volatile int*)&S.win_done[wid & (S.win_cap - 1)], (int)(wid + 1));
  MvmCell* cell = S.fcells + (pos & (unsigned long long)(S.nslots - 1));
  for (int spin = 0; ld_acq64(&cell->seq) != pos; ++spin) {
    if (spin > (1 << 22)) { mvm_fail(S, MVM_ERR_QUEUE_OVERFLOW, slot, -1, (int)pos); return; }
    __nanosleep(32);
  }
  cell->slot = slot;
  st_rel64(&cell->seq, pos + 1);
  red_add(&S.cnt->done, 1);                                                              // counted last; posted
}

__device__ void mvm_worker(const MvmDevState& Sparam, MvmWatch& t0, MvmShared& sh) {
  const MvmDevState S = Sparam;   // a by-value copy: everything is inlined into this worker, so the fields live in registers instead of a generic load through the parameter's address at every use (dozens per hop)
  unsigned backoff = 0;
  if (threadIdx.x == 0) sh.part = mvm_sched_part(S);
  int owed = -1;   // the task whose completion bookkeeping (mvm_post_task) this block still owes; every thread tracks it, thread 32 pays it during the pop (or, for a continuation, while the next descriptor loads)
  unsigned owed_wid = 0;   // its window id, from the descriptor the block already holds
  long long ticket = -1;  // FIFO mode: lane 0's ticket on the ready ring (-1: none held); a ticket taken while idle stays valid until its cell fills
  int next = -1;   // continuation chosen by the task that just ran (mvm_run_task): run it without a pop
  __syncthreads();
  const int part = sh.part;
#ifdef MVM_PROF_GAP
  if (threadIdx.x == 0) sh.pg_valid = 0;
#endif
  for (;;) {
    int cmd;
    int errv = (threadIdx.x == 16) ? *(volatile int*)&S.cnt->err_code : 0;   // read overlapped with the pop / descriptor load (no separate round trip)
    if (next >= 0) {
      cmd = next;
      if (threadIdx.x == 0) sh.how = MVM_TRACE_CONT;
      if (threadIdx.x < MVM_TASK_WORDS) { ((int*)&sh.d)[threadIdx.x] = ((const int*)(S.tasks + cmd))[threadIdx.x]; MVM_PG_SUB(3, 6); }
      else if (threadIdx.x == 32 && owed >= 0) { mvm_post_task(S, owed, owed_wid); MVM_PG_SUB(4, 6); }
      owed = -1;
      if (threadIdx.x == 16) { sh.err = errv; MVM_PG_SUB(5, 6); }
      __syncthreads();   // the descriptor and the previous task's bookkeeping are complete before the op
    } else {
      if (threadIdx.x < 32) {
        int how = 0, dw, r;
        if (S.fifo_only) r = mvm_fifo_pop(S, t0, ticket, dw);
        else {
          r = mvm_pop(S, mvm_now(), part, how, dw);
          if (r < 0 && threadIdx.x == 0 && mvm_should_exit(S, t0)) r = MVM_POP_EXIT;
        }
        if (threadIdx.x == 0) { sh.cmd = r; sh.how = how; }
        if (r >= 0 && threadIdx.x < MVM_TASK_WORDS) ((int*)&sh.d)[threadIdx.x] = dw;
      } else if (threadIdx.x == 32 && owed >= 0) mvm_post_task(S, owed, owed_wid);   // overlapped with the pop above
      owed = -1;
      if (threadIdx.x == 16) sh.err = errv;
      __syncthreads();
      cmd = sh.cmd;
      __syncthreads();  // sh.cmd is rewritten next iteration
      if (cmd == MVM_POP_EXIT) return;
      if (cmd < 0) {
        if (threadIdx.x < 32) __nanosleep(backoff);
        backoff = backoff ? min(backoff * 2, 512u) : 32u;
        continue;
      }
    }
    backoff = 0;
    mvm_run_task(S, cmd, sh);
    owed = cmd; owed_wid = (unsigned)sh.d.reserved0;   // the descriptor still in shared memory describes the task that just ran
    next = sh.cont;
    __syncthreads();   // every thread has read sh.cont before the next task's thread 0 resets it
  }
}

// ---------------- admitter (block 0) ----------------
// A bundle record = one "window". Resources it needs before admission, all bounded: nt free slots, ne successor-ring words,
// one window-table entry, nt of the live window. Successor words and the window entry return in window order, once every
// task of the window has completed (a task blocked on a version pins its window's successor words).
struct MvmAdmitShared {
  int action, hdr[MVM_REC_HDR];
  unsigned long long fhead;                      // next free-ring position to pop
  unsigned win_next, win_oldest, succ_scan, succ_used, succ_free;
};
#define MVM_ADM_IDLE 0
#define MVM_ADM_REC  1
#define MVM_ADM_FINAL 2
#define MVM_ADM_EXIT 3

// Two reclaim pointers, both advanced only here. win_oldest = oldest window with a task still alive: bounds the window table.
// succ_scan walks the successor ring's regions in order: it skips windows with no edges (they own no words) and passes a
// region only when all of its tasks completed, so a task parked on a version pins only the words of its own window.
__device__ void mvm_reclaim_windows(const MvmDevState& S, MvmAdmitShared& A) {  // thread 0
  while (A.win_oldest != A.win_next && ld_acq(&S.wins[A.win_oldest & (S.win_cap - 1)].remaining) == 0) ++A.win_oldest;
  while (A.succ_scan != A.win_next) {
    MvmWin* w = S.wins + (A.succ_scan & (S.win_cap - 1));
    if (w->ne != 0) {
      if (ld_acq(&w->remaining) != 0) break;
      A.succ_free = w->succ_end;
    }
    ++A.succ_scan;
  }
}

// Admission of one bundle record. The record is first bulk-copied from the host-mapped ring into device memory (`S.stage`, wide
// coalesced loads: PCIe reads are latency-bound, so one thread chasing one word at a time cost ~1.5 us per word), unless the host already wrote it there (S.prestaged);
// every later pass reads the staged copy from L2. The successor words are written first, then the accounting, then the descriptors four per thread per round, each task
// released as soon as its descriptor is visible, so workers run while the rest of the record is still being decoded.
__device__ void mvm_admit_record(const MvmDevState& S, MvmWatch& t0, MvmAdmitShared& A, unsigned long long pos) {
  const size_t roff = (size_t)(pos % (unsigned long long)S.ring_words);
  const int nt = A.hdr[2], ne = A.hdr[3], na = A.hdr[4];
  const int* rt = S.stage + roff + MVM_REC_HDR;
  const int* re = rt + (size_t)nt * MVM_TASK_WORDS;
  const int* ra = re + ne;
  if (threadIdx.x == 0) {
    int code = 0;
    if (nt < 1 || ne < 0 || na < 0 || (unsigned)A.hdr[5] != A.win_next) code = MVM_ERR_BAD_RECORD;
    else if (nt > S.nslots) code = MVM_ERR_SLOTS;
    else if (nt > S.live_cap) code = MVM_ERR_BATCH_TOO_LARGE;
    else if ((unsigned)ne > S.succ_cap) code = MVM_ERR_POOL;
    if (code) { mvm_fail(S, code, -1, nt, A.hdr[5]); A.action = MVM_ADM_EXIT; }
    else {
      // Backpressure: wait for every bounded resource. Parked (version-waiting) tasks have their own budget of live_cap: they do not
      // consume the window the publishing bundle needs. More than live_cap parked is reported as PARK_STALL after the stall timeout.
      for (;;) {
        mvm_reclaim_windows(S, A);
        bool room = A.win_next - A.win_oldest < (unsigned)S.win_cap && (A.succ_used + (unsigned)ne - A.succ_free) <= S.succ_cap &&
                    ld_acq(&S.cnt->admitted) - ld_acq(&S.cnt->done) - ld_acq(&S.cnt->parked) + nt <= S.live_cap && ld_acq(&S.cnt->parked) <= S.live_cap &&
                    ld_acq64(&S.cnt->ftail) - A.fhead >= (unsigned long long)nt;
        if (room) break;
        if (*(volatile int*)&S.cnt->err_code || mvm_stalled(S, t0, false)) {
          mvm_fail(S, ld_acq(&S.cnt->parked) > S.live_cap ? MVM_ERR_PARK_STALL : MVM_ERR_TIMEOUT, -1, ld_acq(&S.cnt->admitted), ld_acq(&S.cnt->done));
          A.action = MVM_ADM_EXIT; break;
        }
#ifndef MVM_DIST_NO_BP_POLL  // (negative test: tests/dist backpressure suite must then fail)
        mvm_dist_poll_strided(S, 0, 1);  // MVM_DIST_HOOK: remote publishes must keep flowing while admission waits for room
#endif
        __nanosleep(256);
      }
    }
  }
  __syncthreads();
  if (A.action == MVM_ADM_EXIT) return;
  const unsigned long long fbase = A.fhead;
  const unsigned sbase = A.succ_used;
  // stage the record: 16-byte uncached loads (the host owns these words until ring_head passes them), several in flight per thread. A prestaged record is already in device memory.
  if (!S.prestaged) {
    const int4* src = reinterpret_cast<const int4*>(S.ring + roff);
    int4* dst = reinterpret_cast<int4*>(S.stage + roff);
    const int nv = A.hdr[1] / 4;
    for (int q = threadIdx.x; q < nv; q += 8 * blockDim.x) {
      int4 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) { const int k = q + u * blockDim.x; if (k < nv) v[u] = __ldcv(src + k); }
#pragma unroll
      for (int u = 0; u < 8; ++u) { const int k = q + u * blockDim.x; if (k < nv) dst[k] = v[u]; }
    }
  }
  // pop nt slots from the free ring (each producer finishes its cell write within a few instructions); four cells per thread in flight. A run that never recycles uses the slots in
  // record order, so a task's slot is simply fbase + its index.
  if (!S.no_recycle) {
    for (int base = threadIdx.x; base < nt; base += 4 * blockDim.x) {
      unsigned long long sq[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) { const int i = base + u * blockDim.x; if (i < nt) sq[u] = mvm_voll(&S.fcells[(fbase + i) & (unsigned long long)(S.nslots - 1)].seq); }
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        const int i = base + u * blockDim.x; if (i >= nt) break;
        const unsigned long long fp = fbase + i;
        MvmCell* cell = S.fcells + (fp & (unsigned long long)(S.nslots - 1));
        while (sq[u] != fp + 1) { if (*(volatile int*)&S.cnt->err_code) break; __nanosleep(32); sq[u] = mvm_voll(&cell->seq); }   // a worker that failed in mvm_post_task never publishes
        asm volatile("fence.acq_rel.gpu;" ::: "memory");   // acquire: the producer's slot write (before its release of seq) is visible
        S.adm_slots[i] = cell->slot;
        st_rel64(&cell->seq, fp + (unsigned long long)S.nslots);
      }
    }
    __syncthreads();
  }
  const unsigned long long now = S.trace_on ? mvm_now() : 0ull;
  const unsigned wid = A.win_next;
  // An aliased ring (S.succ_alias) is only this record's own successor words, read in place: valid for the first record at ring offset 0 with nothing recycled.
  // Anything else would read another record's words as successors, so it stops the run instead.
  if (S.succ_alias && (roff != 0 || fbase != 0 || sbase != 0 || !S.no_recycle)) {
    if (threadIdx.x == 0) { mvm_fail(S, MVM_ERR_ALIAS_RECORD, -1, (int)roff, (int)sbase); A.action = MVM_ADM_EXIT; }
    return;
  }
  // Successor words first, then the window and admitted accounting: a task released below can finish before its own record has been read, and every successor list it decrements through
  // must already be in place. Eight words per thread in flight; a stage load, then (with recycling) the map from record index to slot. An aliased ring is already in place.
  for (int base = threadIdx.x; !S.succ_alias && base < ne; base += 8 * blockDim.x) {
    int v[8], sl[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) { const int e = base + u * blockDim.x; v[u] = e < ne ? __ldcg(re + e) : 0; }
    if (!S.no_recycle) {
#pragma unroll
      for (int u = 0; u < 8; ++u) { const int e = base + u * blockDim.x; sl[u] = (e < ne && v[u] >= 0 && v[u] < nt) ? __ldcg(S.adm_slots + v[u]) : -1; }
    }
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int e = base + u * blockDim.x; if (e >= ne) break;
      const int t = v[u];
      if (t < 0 || t >= nt) { mvm_fail(S, MVM_ERR_BAD_TASK, S.no_recycle ? (int)fbase : S.adm_slots[0], t, e); S.succ[(sbase + (unsigned)e) & (S.succ_cap - 1)] = S.no_recycle ? (int)fbase : S.adm_slots[0]; continue; }
      S.succ[(sbase + (unsigned)e) & (S.succ_cap - 1)] = S.no_recycle ? (int)(fbase + (unsigned long long)t) : sl[u];
    }
  }
  __threadfence();   // every thread's successor words are visible GPU-wide before the barrier
  __syncthreads();
  if (threadIdx.x == 0) {
    if (!*(volatile int*)&S.cnt->err_code) {
      S.wins[wid & (S.win_cap - 1)].remaining = nt;
      S.wins[wid & (S.win_cap - 1)].succ_end = sbase + (unsigned)ne;
      S.wins[wid & (S.win_cap - 1)].ne = ne;
      __threadfence();
      A.fhead += nt; A.succ_used += ne; A.win_next++;
      atom_add(&S.cnt->admitted, nt);  // before any task of this bundle can be enqueued (done <= admitted always)
    }
  }
  __syncthreads();
  if (*(volatile int*)&S.cnt->err_code) return;
  // Decode descriptors, four per thread per round, each task released as soon as its own descriptor is written: workers start one round after the first load instead of after the whole record.
  // A predecessor from an earlier round can finish before a successor's descriptor is written: its decrement drives S.indeg[slot] negative (it is 0 whenever a slot is free, a task being
  // released exactly when its count reaches 0), and the admitter ADDS the true indegree after the descriptor is visible (a fence, then the add): old + ind == 0 means every predecessor has
  // already finished and this thread releases it; otherwise the last predecessor's decrement (old == 1) does. Exactly one of the two sees zero.
  for (int base = threadIdx.x; base < nt; base += 4 * blockDim.x) {
    int w[4][MVM_TASK_WORDS], slot[4], ind[4], olds[4]; bool valid[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int i = base + u * blockDim.x;
      valid[u] = i < nt;
      if (valid[u]) {
        const int4* p = reinterpret_cast<const int4*>(rt + (size_t)i * MVM_TASK_WORDS);
#pragma unroll
        for (int k = 0; k < MVM_TASK_WORDS / 4; ++k) { const int4 v = __ldcg(p + k); w[u][4 * k] = v.x; w[u][4 * k + 1] = v.y; w[u][4 * k + 2] = v.z; w[u][4 * k + 3] = v.w; }
        slot[u] = S.no_recycle ? (int)(fbase + (unsigned long long)i) : __ldcg(S.adm_slots + i);
      }
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (!valid[u]) continue;
      const int i = base + u * blockDim.x;
      // MvmTaskDesc word order: op, succ_off, succ_cnt, indegree, priority, deadline, args_off, args_words, wait_var,
      // wait_ver, pub_var, pub_ver, workload, flags, r0, r1
      int* wu = w[u];
      const int sl = slot[u];
      const int op = wu[0], so = wu[1], sc = wu[2], pr = wu[4], ao = wu[6], aw = wu[7], wv = wu[8], pv = wu[10];
      ind[u] = wu[3];
      int a[MVM_MAX_ARGS];   // the argument words, fetched in one go when the range is sane (checked again below)
      const bool arg_range = ao >= 0 && aw >= 0 && aw <= MVM_MAX_ARGS && (long long)ao + aw <= na;
#pragma unroll
      for (int j = 0; j < MVM_MAX_ARGS; ++j) a[j] = (arg_range && j < aw) ? __ldcg(ra + ao + j) : 0;
      int code = 0;
      if (mvm_dev_arity(op) < 0) code = MVM_ERR_BAD_OP;
      else if (aw != mvm_dev_arity(op)) code = MVM_ERR_BAD_ARITY;
      else if (!mvm_sched_class_ok(S, wu[12])) code = MVM_ERR_BAD_TASK;
      else if (so < 0 || sc < 0 || (long long)so + sc > ne || !arg_range || ind[u] < 0 || ind[u] > nt || pr < 0 || pr >= MVM_NBUCKETS || wu[5] < -1) code = MVM_ERR_BAD_TASK;
      else if (wv < -1 || wv >= S.num_vars || pv < -1 || pv >= S.num_vars) code = MVM_ERR_BAD_VAR;
      else {
        const uint64_t ty = mvm_dev_types(op);
        for (int j = 0; j < aw && !code; ++j) {
          const int t = (int)((ty >> (3 * j)) & 7u);
          if (t == MVM_A_BUF || t == MVM_A_BUF_OPT) {
            const int bi = a[j];
            if (bi == -1 && t == MVM_A_BUF) code = MVM_ERR_BAD_BUF;
            else if (bi < -1 || bi >= S.nbufs) code = MVM_ERR_BAD_BUF_INDEX;
          }
        }
      }
      if (code) { mvm_fail(S, code, sl, op, i); valid[u] = false; continue; }
      for (int j = 0; j < aw; ++j) S.args[(size_t)sl * MVM_ARG_STRIDE + j] = a[j];
      wu[1] = (int)(sbase + (unsigned)so);       // CSR offset into the successor ring (word sequence number)
      wu[6] = sl * MVM_ARG_STRIDE;               // arguments live inline in the slot
      wu[14] = (int)wid;                         // reserved0 = window id (task completion returns the window credit)
      int4* dst = reinterpret_cast<int4*>(S.tasks + sl);
#pragma unroll
      for (int k = 0; k < MVM_TASK_WORDS / 4; ++k) dst[k] = make_int4(wu[4 * k], wu[4 * k + 1], wu[4 * k + 2], wu[4 * k + 3]);
      S.state[sl] = 0; S.park_next[sl] = -1;
      if (S.trace_on) { MvmTrace& t = S.trace[sl]; t.admit_ns = now; t.ready_ns = t.start_ns = t.end_ns = 0; t.flags = 0; t.exec_count = 0; t.ver0 = t.ver1 = -1; }
      mvm_sched_admit_task(S, sl);                // SCHED: cost metadata, publisher classes (reads the args and descriptor this thread just wrote)
    }
    __threadfence();                              // the descriptors, arguments and metadata this thread wrote are visible before the indegree adds that can make a task runnable
#ifndef MVM_BREAK_CHUNK_RELEASE
#pragma unroll
    for (int u = 0; u < 4; ++u) if (valid[u]) olds[u] = atom_add(S.indeg + slot[u], ind[u]);
#else   // negative control (tests/litmus/run_litmus.sh): store instead of add loses the decrements of early-finished predecessors
#pragma unroll
    for (int u = 0; u < 4; ++u) if (valid[u]) { S.indeg[slot[u]] = ind[u]; olds[u] = 0; }
#endif
#pragma unroll
    for (int u = 0; u < 4; ++u) if (valid[u] && olds[u] + ind[u] == 0) mvm_release_task(S, slot[u], w[u][8], w[u][9]);   // wait_var, wait_ver
  }
  __syncthreads();
}

__device__ void mvm_admitter(const MvmDevState& S, MvmWatch& t0, MvmAdmitShared& A) {
  unsigned long long pos = 0;
  if (threadIdx.x == 0) { A.fhead = 0; A.win_next = A.win_oldest = A.succ_scan = A.succ_used = A.succ_free = 0; }
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0) {
      A.action = MVM_ADM_IDLE;
      if (*(volatile int*)&S.cnt->err_code || mvm_stalled(S, t0)) {
        mvm_fail(S, ld_acq(&S.cnt->parked) > S.live_cap ? MVM_ERR_PARK_STALL : MVM_ERR_TIMEOUT, -1, ld_acq(&S.cnt->admitted), ld_acq(&S.cnt->done));
        A.action = MVM_ADM_EXIT;
      } else {
        unsigned closed = ld_acq_sys(S.closed);                    // BEFORE the doorbell: closed => doorbell is final
        unsigned long long tail = ld_acq_sys64(S.doorbell);        // acquire: record words below are visible
        if (pos < tail) {
          const int* rec = S.ring + (size_t)(pos % (unsigned long long)S.ring_words);
          for (int j = 0; j < MVM_REC_HDR; ++j) A.hdr[j] = ld_sys(rec + j);
          unsigned long long left = (unsigned long long)S.ring_words - pos % (unsigned long long)S.ring_words;
          bool ok = A.hdr[0] == MVM_REC_MAGIC && A.hdr[1] >= MVM_REC_HDR && A.hdr[1] % 8 == 0 &&
                    (unsigned long long)A.hdr[1] <= left && pos + A.hdr[1] <= tail &&
                    (A.hdr[6] == MVM_REC_PAD || (A.hdr[6] == MVM_REC_BUNDLE && A.hdr[1] == ((MVM_REC_HDR + A.hdr[2] * MVM_TASK_WORDS + A.hdr[3] + A.hdr[4] + 7) & ~7)));
          if (!ok) { mvm_fail(S, MVM_ERR_BAD_RECORD, -1, (int)pos, A.hdr[1]); A.action = MVM_ADM_EXIT; }
          else A.action = MVM_ADM_REC;
        } else if (closed) A.action = MVM_ADM_FINAL;
      }
    }
    __syncthreads();
    int action = A.action;
    __syncthreads();
    if (action == MVM_ADM_EXIT) return;
    if (action == MVM_ADM_FINAL) { if (threadIdx.x == 0) st_rel(&S.cnt->admit_final, 1); mvm_dist_pump(S, t0); return; }  // MVM_DIST_HOOK
    if (action == MVM_ADM_IDLE) { mvm_dist_poll(S); __nanosleep(256); continue; }  // MVM_DIST_HOOK
    if (A.hdr[6] == MVM_REC_BUNDLE) mvm_admit_record(S, t0, A, pos);
    __syncthreads();
    if (A.action == MVM_ADM_EXIT) return;
    pos += (unsigned long long)A.hdr[1];
    if (threadIdx.x == 0) st_rel_sys64(S.ring_head, pos);  // ring space returns to the host only after the copy
    __syncthreads();
  }
}

__global__ void __launch_bounds__(MVM_THREADS) mvm_init_kernel(MvmDevState S) {
  size_t nq = S.fifo_only ? 1 : (size_t)S.sp.nc * MVM_NBUCKETS, n = nq * S.rcap;   // SCHED: one ring set per class; FIFO mode: one ring
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n; i += stride) { S.cells[i].seq = i % (size_t)S.rcap; S.cells[i].slot = -1; }
  for (size_t i = tid; i < (size_t)S.nslots; i += stride) { S.fcells[i].seq = i + 1; S.fcells[i].slot = (int)i; }
  for (size_t i = tid; i < (size_t)S.num_vars; i += stride) S.park_head[i] = -1;
  if (tid == 0) S.cnt->ftail = (unsigned long long)S.nslots;
}

__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_exec_kernel(const __grid_constant__ MvmDevState S) {
  if (S.warm_only) return;
  __shared__ MvmShared sh;
  __shared__ MvmAdmitShared adm;
  MvmWatch t0{mvm_now(), 0, -1, -1};
  auto pow2 = [](unsigned long long v) { return v >= 2 && !(v & (v - 1)); };
  if (gridDim.x < 2 + S.sp.on || !pow2(S.live_cap) || S.rcap != 4 * S.live_cap || !pow2(S.nslots) || !pow2(S.succ_cap) || !pow2(S.win_cap)) {
    if (threadIdx.x == 0 && blockIdx.x == 0) mvm_fail(S, MVM_ERR_BAD_CONFIG, -1, gridDim.x, S.live_cap);
    return;
  }
  __shared__ MvmPolShared pol;
  if (blockIdx.x == 0) mvm_admitter(S, t0, adm);
  else if (S.sp.on && blockIdx.x == 1) mvm_policy(S, t0, pol);  // SCHED: block 1 is the policy block when the scheduler is on
  else mvm_worker(S, t0, sh);
}

#include "mvm_static.cuh"

// ---------------- initial values ----------------
// Counter-based generator, bit-identical to runtime/initgen.py: draw j of seed s is SplitMix64 at counter j.
__host__ __device__ inline unsigned long long mvm_splitmix(unsigned long long seed, unsigned long long j) {
  unsigned long long z = seed + (j + 1) * 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__host__ __device__ inline float mvm_u24(unsigned long long seed, unsigned long long j) { return (float)(unsigned)(mvm_splitmix(seed, j) >> 40) * (1.0f / 16777216.0f); }
// Every step is one correctly rounded f32 operation (no fused multiply-add), like the reference.
__host__ __device__ inline float mvm_seeded_value(unsigned long long seed, unsigned long long i, int draws, float bias, float coef) {
  float acc;
  if (draws == 1) acc = mvm_u24(seed, i);
  else {
    unsigned long long j = i * (unsigned long long)draws;
#ifdef __CUDA_ARCH__
    acc = __fadd_rn(__fadd_rn(mvm_u24(seed, j), mvm_u24(seed, j + 1)), __fadd_rn(mvm_u24(seed, j + 2), mvm_u24(seed, j + 3)));
    return __fmul_rn(__fsub_rn(acc, bias), coef);
#else
    volatile float a = mvm_u24(seed, j) + mvm_u24(seed, j + 1), b = mvm_u24(seed, j + 2) + mvm_u24(seed, j + 3);
    acc = a + b;
#endif
  }
#ifdef __CUDA_ARCH__
  return __fmul_rn(__fsub_rn(acc, bias), coef);
#else
  volatile float d = acc - bias; return d * coef;
#endif
}
__global__ void mvm_seeded_fill_kernel(float* p, unsigned long long n, unsigned long long seed, int draws, float bias, float coef) {
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (unsigned long long)gridDim.x * blockDim.x)
    p[i] = mvm_seeded_value(seed, i, draws, bias, coef);
}
#endif  // __CUDACC__
