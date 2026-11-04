// KVM distributed execution: device side. No NCCL, no host on the critical path (intra-node), one host proxy thread per device
// for inter-node. Full design, modeled-vs-built status and scaling math: DISTRIBUTED.md.
//
// ===== Model =====
//  N devices, each running its own executor instance (one persistent kernel) over its shard of the task graph.
//  A version variable is a monotone counter owned by ONE device (the one whose tasks wait on it). The ABI is unchanged: a
//  consumer is an ordinary task with wait_var/wait_ver. A producer on another device publishes by writing the owner's INBOX
//  (u32[num_vars] in the owner's device memory; a KvmBuf table entry on the producer, so no new descriptor field):
//      inbox[var] = ver         st.release.sys, after fence.sc.sys, after every store of the task (see below)
//  The owner's admitter block (block 0, which is otherwise idle between bundles; see KVM_DIST_HOOK in kvm_exec.cuh) polls the
//  inbox variables listed in KvmDistDev::in_vars and performs the LOCAL publish: version[var] = inbox value, then wakes parked
//  tasks exactly as kvm_publish does. Workers never poll; a remote waiter occupies no block (it is parked like any waiter).
//
// ===== Memory-ordering argument (cross-device) =====
//  Producer task T on device A: all threads store tile data into the consumer's memory (peer pointer); __syncthreads();
//  thread 0: fence.sc.sys; st.release.sys inbox[var] = ver. bar.sync + fence.sc.sys is cumulative over every thread's stores
//  (same argument as the executor's gpu-scope publish, widened to system scope), so the data stores happen-before the release.
//  Owner poller P: ld.acquire.sys inbox[var] reads ver => synchronizes-with T's release. P then st.release.gpu version[var]
//  (after fence.sc.gpu) and wakes the waiters. Waiter W (ld.acquire.gpu version, then bar.sync) observes P's release, hence
//  T's data stores: base causality order is transitive across the sys-scope and gpu-scope edges. A waiter that sees `ver`
//  sees ALL data of T and of everything T's block had observed. (Peer stores over NVLink/PCIe are posted writes; the sys fence
//  is what drains them, which is why it is mandatory and not an optimisation target.)
//  Inbox variables are single-writer chains with strictly increasing versions, so st.release (not a read-modify-write) is
//  enough: no PCIe atomics are required on peer memory.
//  Host side (proxy): x86-64 TSO needs only the compiler/atomic acquire on the queue slot sequence; aarch64 (Grace) needs the
//  hardware acquire/release that std::atomic emits. The proxy uses std::atomic with acquire/release throughout, never volatile.
//  The inter-node send queue lives in mapped pinned host memory (slots and the host-written head); its ticket counter is in
//  DEVICE memory, because atomics on pinned host memory need PCIe AtomicOps (cudaDevAttrHostNativeAtomicSupported is 0 on x86
//  hosts). The device writes payload with plain stores, then publishes
//  the slot sequence with st.release.sys; the proxy loads it with acquire, so payload reads are ordered (formally sound, unlike
//  host-side cudaMemcpy of device memory while a kernel runs). Receive: cudaMemcpyAsync(data) then cudaMemcpyAsync(inbox word)
//  in ONE stream; stream order completes the data copy before the inbox write starts (engine-level ordering, see DISTRIBUTED.md).
//
// ===== Negative test =====
//  -DKVM_DIST_BREAK_ORDER makes put_signal/signal-with-data publish BEFORE the data stores (and delays the stores). The
//  harness must observe stale data and FAIL (tests/dist/test_dist.cu --expect-fail).
//
// (No #pragma once: it is included twice by kvm_exec.cuh, each phase guarded by its own macro.)
// Included twice by kvm_exec.cuh: phase 1 (types, errors, primitives) before the operator packs, phase 2 (inbox pump) after
// kvm_publish, because the pump needs the ready-queue functions.
#ifndef KVM_DIST_PUMP_PHASE
#ifndef KVM_DIST_PHASE1
#define KVM_DIST_PHASE1
#include "../dist/wire.hpp"

struct KvmDistCtl {          // mapped pinned host memory, written by the proxy, read by the pump
  volatile int fail_code;    // 0 or a KVM_ERR_DIST_* code; the pump turns it into a device error
  volatile int fail_arg;
  volatile unsigned heartbeat;   // proxy increments every loop; 0 = no proxy configured
  unsigned last_hb;              // pump bookkeeping (device-written, host never reads)
  unsigned long long last_ns;
  unsigned long long dead_ns;    // 0 = do not watch the heartbeat
};
struct KvmDistDev {          // device-resident (or mapped), pointed to by KvmDevState::dist
  int rank, world, n_in, pad;
  const int* in_vars;        // local version variables fed by remote publishers
  unsigned* inbox;           // u32[num_vars], remote publishers store here
  KvmDistCtl* ctl;           // may be null (no proxy)
};

#ifdef __CUDACC__
__device__ __forceinline__ void kvm_fence_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ void st_rel_sys32(unsigned* p, unsigned v) { asm volatile("st.release.sys.global.b32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }

// Publish (var, ver) to a remote inbox. Caller: every thread's data stores precede a __syncthreads(); thread 0 only.
__device__ __forceinline__ void kvm_remote_publish(unsigned* inbox, int var, int ver) {
  kvm_fence_sys();
  st_rel_sys32(inbox + var, (unsigned)ver);
}

// Tile validation: rows,cols >= 1, ld >= cols, [off, off + (rows-1)*ld + cols) inside the view. Reports KVM_ERR_BOUNDS.
__device__ __forceinline__ bool kvm_tile_ok(const KvmCtx& c, const KvmView& v, int off, int rows, int cols, int ld) {
  if (v.p == nullptr) return false;
  bool ok = rows >= 1 && cols >= 1 && ld >= cols && off >= 0 &&
            (long long)off + (long long)(rows - 1) * ld + cols <= (long long)v.rows * v.cols;
  if (!ok) c.fail(KVM_ERR_BOUNDS, off, rows * 1000003 + cols);
  return ok;
}

// Block-cooperative tile copy; 16-byte vectors when every pointer/stride is aligned.
__device__ __forceinline__ void kvm_tile_copy(float* d, int dld, const float* s, int sld, int rows, int cols) {
  const bool vec = !(((uintptr_t)d | (uintptr_t)s) & 15) && !((cols | dld | sld) & 3);
  if (vec) {
    const int c4 = cols >> 2;
    const long long total = (long long)rows * c4;
    for (long long i = threadIdx.x; i < total; i += blockDim.x) {
      const int r = (int)(i / c4), k = (int)(i - (long long)r * c4);
      ((float4*)(d + (size_t)r * dld))[k] = ((const float4*)(s + (size_t)r * sld))[k];
    }
  } else {
    const long long total = (long long)rows * cols;
    for (long long i = threadIdx.x; i < total; i += blockDim.x) {
      const int r = (int)(i / cols), k = (int)(i - (long long)r * cols);
      d[(size_t)r * dld + k] = s[(size_t)r * sld + k];
    }
  }
}

// Inbox/flag view check: var inside the mailbox buffer.
__device__ __forceinline__ bool kvm_flag_ok(const KvmCtx& c, const KvmView& f, int var, int ver) {
  if (f.p == nullptr) return false;
  if (var < 0 || (long long)var >= (long long)f.rows * f.cols || ver < 1) { c.fail(KVM_ERR_DIST_BAD_SIGNAL, var, ver); return false; }
  return true;
}
#endif  // __CUDACC__
#endif  // KVM_DIST_PHASE1

#elif !defined(KVM_DIST_PHASE2)
#define KVM_DIST_PHASE2
// ===================== phase 2: the inbox pump (needs kvm_enqueue / kvm_park) =====================

// Mirror of the tail of kvm_publish: after version[var] advanced, wake every parked task whose wait is now satisfied.
// If kvm_publish's wake loop changes, change this too (the litmus tests in tests/dist exercise both).
__device__ __forceinline__ void kvm_dist_wake(const KvmDevState& S, int var) {
  __threadfence();                                              // version store -> fence -> list exchange (Dekker, see header)
  int n = atom_exch(S.park_head + var, -1);
  while (n >= 0) {
    int next = S.park_next[n];
    int need = S.tasks[n].wait_ver;
    if (ld_acq(S.version + var) >= need) {
      if (atom_cas(S.state + n, 1, 2) == 1) {
#if __has_include("kvm_sched.cuh")
        kvm_sched_unpark(S, n);                       // feat/sched keeps a count of parked waiters: release it like kvm_publish does
#endif
        kvm_enqueue(S, n);
      }
    } else kvm_park(S, n, var, need);
    n = next;
  }
}

// Poll the inbox variables i = tid, tid+nthreads, ... Effects are version stores and wakeups. Single writer per variable (this
// poller), so a plain release store is race-free. Callers: all threads of the admitter block (tid = threadIdx.x), or thread 0
// alone (0, 1) from inside the admission backpressure wait, which is thread-0 code: without that call a live window full of
// tasks parked on remote variables would wait for the very signals the blocked admitter never gets to publish.
__device__ __forceinline__ void kvm_dist_poll_strided(const KvmDevState& S, int tid, int nthreads) {
  const KvmDistDev* D = S.dist;
  if (D == nullptr) return;
  for (int i = tid; i < D->n_in; i += nthreads) {
    const int v = D->in_vars[i];
    const int got = (int)ld_acq_sys(D->inbox + v);              // acquire.sys: pairs with the remote release (header)
    if (got > ld_acq(S.version + v)) {
      __threadfence();                                          // sys acquire -> fence.sc.gpu -> release.gpu (transitivity)
      st_rel(S.version + v, got);
      kvm_dist_wake(S, v);
    }
  }
  if (tid == 0 && D->ctl != nullptr) {
    int code = *(volatile int*)&D->ctl->fail_code;
    if (code) kvm_fail(S, code, -1, *(volatile int*)&D->ctl->fail_arg, D->rank);
    else if (D->ctl->dead_ns && *(volatile unsigned*)&D->ctl->heartbeat) {         // a stalled proxy would hang every remote wait
      const unsigned hb = *(volatile unsigned*)&D->ctl->heartbeat;
      const unsigned long long now = kvm_now();
      if (hb != D->ctl->last_hb) { D->ctl->last_hb = hb; D->ctl->last_ns = now; }
      else if (D->ctl->last_ns && now - D->ctl->last_ns > D->ctl->dead_ns) kvm_fail(S, KVM_ERR_DIST_PROXY_DEAD, -1, (int)((now - D->ctl->last_ns) / 1000000ull), D->rank);
    }
  }
}
__device__ __forceinline__ void kvm_dist_poll(const KvmDevState& S) { kvm_dist_poll_strided(S, threadIdx.x, blockDim.x); }

// Called by the admitter block after the last bundle: keep pumping until the device is done (or failed / timed out).
__device__ void kvm_dist_pump(const KvmDevState& S, unsigned long long t0) {
  if (S.dist == nullptr) return;
  __shared__ int stop;
  for (;;) {
    if (threadIdx.x == 0) {
      stop = 0;
      if (*(volatile int*)&S.cnt->err_code) stop = 1;
      else if (ld_acq(&S.cnt->done) >= ld_acq(&S.cnt->admitted)) stop = 1;
      else if (kvm_now() - t0 > S.timeout_ns) { kvm_fail(S, KVM_ERR_TIMEOUT, -1, ld_acq(&S.cnt->admitted), ld_acq(&S.cnt->done)); stop = 1; }
    }
    __syncthreads();
    const int s = stop;
    __syncthreads();
    if (s) return;
    kvm_dist_poll(S);
    __nanosleep(128);
  }
}
#endif
