#pragma once
// KVM static schedules, device side: one persistent kernel whose blocks each run a program compiled before launch (kvm_static.h).
//
//   * A worker (one block of KVM_THREADS threads) executes its records in order. Records are fully resolved: op, argument words and buffer views
//     are read from shared memory, where the next record is copied (cp.async) while the current one runs.
//   * Between two records of one worker the only synchronization is __syncthreads(). A dependency across workers is a wait on the
//     publishing worker's watermark: thread 0 polls progress[w] with an acquire load, then a barrier hands the data to the block.
//   * A record that another worker waits for publishes: every warp fences (fence.sc.gpu makes ITS stores visible across SMs, kvm_exec.cuh's
//     rule), a barrier, thread 0's release store of the watermark. Nothing else costs an atomic or a queue.
// Memory-ordering, with the waiter's thread 0 and the publisher's thread 0 as the anchors: the waiter's ld.acquire.gpu synchronizes with the
// publisher's st.release.gpu, whose fence-then-barrier history covers every store of every record the publisher ran before it; the
// waiter's barrier extends that to all of its threads (kvm_exec.cuh's argument, unchanged).
#include "kvm_static.h"
#include "kvm_fused_mlp.cuh"

// The dynamic shared-memory arena every op takes from offset 0 (the same declaration as kvm_ops_gpt.cuh's, repeated so a build without the gpt pack, like tests/gpu/build_core.sh, still has it).
extern __shared__ __align__(128) unsigned char kvm_gpt_smem[];

struct KvmStaticDev {
  const KvmSRec* recs;            // every worker's records, concatenated
  const int* offs;                // [nworkers + 1]: worker w runs recs[offs[w] .. offs[w+1])
  int* progress;                  // [nworkers * KVM_S_LINE_WORDS]: per worker a monotone watermark on its own 128-byte line
  int nworkers, trace_on;
  int nactive;                    // workers with a program (the rest exit at once, or spin with spin_idle)
  int spin_idle;                  // 1: a worker with no program polls `finished` until every active worker is done (keeps the SMs busy for the clock governor)
  int* finished;                  // [1] workers that have completed their program
  unsigned long long timeout_ns;  // stall limit of one wait: a worker that waits this long fails the run (a missing publisher is a bug)
  KvmCounters* cnt;
  KvmHostErr* herr;
  unsigned long long* trace;      // [nrecs * KVM_S_TRACE_WORDS]: start, waits done, op done, published (nullptr when off)
  const KvmFusedDesc* fused;      // the fused runs of this bundle (kvm_fuse.h); nullptr when it has none
  long long* fuse_phases;         // [nworkers * KVM_FMLP_NPHASES]: per worker, cycles in each phase of its slices (--fuse-phases; nullptr when off)
};
#define KVM_S_LINE_WORDS 32         // one watermark per 128-byte line: one writer, a few pollers, no false sharing
#define KVM_S_TRACE_WORDS 4

#ifdef __CUDACC__
struct KvmSShared { KvmSRec rec[2]; int stop; long long t_phase; float smem[KVM_SMEM_FLOATS]; };   // two record slots: current and being fetched

// Copy one 512-byte record into shared memory: one 16-byte cp.async per lane of warp 0, committed as one group per lane.
__device__ __forceinline__ void kvm_s_fetch(const KvmSRec* g, KvmSRec* s) {
  if (threadIdx.x < 32) {
    const unsigned d = (unsigned)__cvta_generic_to_shared((char*)s + 16 * threadIdx.x);
    const char* src = (const char*)g + 16 * threadIdx.x;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
}
__device__ __forceinline__ void kvm_s_fetch_wait() { if (threadIdx.x < 32) asm volatile("cp.async.wait_all;" ::: "memory"); }

// Thread 0: poll one watermark until it reaches `need`. Returns false on a stall or a failure elsewhere.
__device__ __forceinline__ bool kvm_s_await(const KvmStaticDev& S, int worker, int need, int task) {
  int* p = S.progress + (size_t)worker * KVM_S_LINE_WORDS;
  unsigned long long t0 = 0;
  for (unsigned spin = 0;; ++spin) {
    if (ld_acq(p) >= need) return true;
    if ((spin & 0xfff) == 0xfff) {     // every ~1-2 ms of polling
      if (*(volatile int*)&S.cnt->err_code) return false;
      const unsigned long long n = kvm_now();
      if (!t0) t0 = n;
      else if (n - t0 > S.timeout_ns) { kvm_fail(S.cnt, S.herr, KVM_ERR_TIMEOUT, task, worker, need); return false; }
    }
  }
}

// The fused ops' bodies as functions of their own, not inlined into the worker loop: the loop's live state then stays out of their register allocation, and the GEMMs' accumulators
// do not spill around it (R-33: with them inlined the static kernel ran 255 registers and a 3.6 KB stack frame; the same body on the bed, alone, 64 bytes).
__device__ __noinline__ void kvm_s_fmlp_slice(int sc, const int32_t* ids, const int32_t* tgt, float* arena, long long* ph, long long* ph_last, KvmCtx* c) {
  kvm_fmlp::slice_run(sc, ids, tgt, nullptr, arena, kvm_fmlp::PhaseTimer{ph, ph_last}, [c](int code, int v) { c->fail(code, v); });
}
__device__ __noinline__ void kvm_s_fmlp_sgd(int owner, float* loss, float* arena) { kvm_fmlp::sgd_run(owner, loss, arena); }

__device__ void kvm_static_worker(const KvmStaticDev& Sref, KvmSShared& sh) {
  const KvmStaticDev S = Sref;   // a register copy of the launch state: fields then read from registers, not through the __grid_constant__ reference (the same copy measured x1.04-1.09 on the dynamic worker)
  const int w = blockIdx.x;
  if (w >= S.nworkers) return;     // (the warm-up launch: nworkers = 0, every block returns at once, the code is loaded)
  const int first = S.offs[w], n = S.offs[w + 1] - first;
  if (n <= 0) {
    if (S.spin_idle && threadIdx.x == 0) { while (ld_acq(S.finished) < S.nactive && !*(volatile int*)&S.cnt->err_code) __nanosleep(1000); }
    return;
  }
  const KvmSRec* prog = S.recs + first;
  if (threadIdx.x == 0) sh.stop = 0;
  kvm_s_fetch(prog, &sh.rec[0]);
  kvm_s_fetch_wait();
  __syncthreads();
  for (int pc = 0; pc < n; ++pc) {
    const KvmSRec* r = &sh.rec[pc & 1];
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * KVM_S_TRACE_WORDS] = kvm_now();
    if (pc + 1 < n) kvm_s_fetch(prog + pc + 1, &sh.rec[(pc + 1) & 1]);   // the next record is copied beside this one's waits and op body
    if (r->op == KVM_OP_fmlp_slice || r->op == KVM_OP_fmlp_sgd) kvm_fmlp::prefetch_desc(S.fused + r->args[0], reinterpret_cast<float*>(kvm_gpt_smem));   // a fused op's descriptor: its round trip overlaps this record's waits
    const int nw = r->nwait;
    if (nw) {   // only a record with waits needs the hand-over barrier (a plain chain continues directly)
      if (threadIdx.x < nw) {   // one lane of warp 0 per wait: a fan-in of up to KVM_S_WAITS costs one poll, not one per source
        if (!kvm_s_await(S, r->waits[threadIdx.x].worker, r->waits[threadIdx.x].need, r->task)) sh.stop = 1;
      }
      __syncthreads();    // every polling lane's acquire is every thread's (causality through the barrier)
    }
    if (sh.stop) { kvm_s_fetch_wait(); return; }
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * KVM_S_TRACE_WORDS + 1] = kvm_now();
    const int flags = r->flags;
    if (!(flags & KVM_S_WAITONLY)) {
      KvmCtx c{r->bufs, r->nbufs, r->task, S.cnt, S.herr, sh.smem};
      float* const arena = reinterpret_cast<float*>(kvm_gpt_smem);
      if (r->op == KVM_OP_fmlp_slice) kvm_s_fmlp_slice(r->args[1], reinterpret_cast<const int32_t*>(r->bufs[r->args[2]].p), reinterpret_cast<const int32_t*>(r->bufs[r->args[3]].p), arena, S.fuse_phases ? S.fuse_phases + (size_t)w * KVM_FMLP_NPHASES : nullptr, &sh.t_phase, &c);
      else if (r->op == KVM_OP_fmlp_sgd) kvm_s_fmlp_sgd(r->args[1], reinterpret_cast<float*>(r->bufs[r->args[2]].p), arena);
      else kvm_dispatch(r->op, c, r->args);
    }
    if (threadIdx.x == 0 && ((pc & 63) == 63) && *(volatile int*)&S.cnt->err_code) sh.stop = 1;   // a failure elsewhere ends the run within 64 records
    if (flags & KVM_S_PUBLISH) __threadfence();    // each warp makes its own op stores visible GPU-wide (kvm_exec.cuh, kvm_run_task)
    kvm_s_fetch_wait();                             // the next record has landed before the barrier that exposes it
    __syncthreads();                                // also exposes sh.stop
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * KVM_S_TRACE_WORDS + 2] = kvm_now();
    if ((flags & KVM_S_PUBLISH) && threadIdx.x == 0) st_rel(S.progress + (size_t)w * KVM_S_LINE_WORDS, r->pos1);   // release: every warp's fence finished before the barrier, so no further fence is needed
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * KVM_S_TRACE_WORDS + 3] = kvm_now();
  }
  if (threadIdx.x == 0) atomicAdd(S.finished, 1);
}

__global__ void __launch_bounds__(KVM_THREADS, KVM_MIN_BLOCKS) kvm_static_kernel(const __grid_constant__ KvmStaticDev S) {
  __shared__ KvmSShared sh;
  kvm_static_worker(S, sh);
}
#endif  // __CUDACC__
