#pragma once
// KVM scheduling policy, device side. Included by kvm_exec.cuh at the "ready queue" section (after KvmDevState, the PTX
// primitives and kvm_fail). See kvm_sched.h for the data and the vocabulary.
//
// ===== What the scheduler is =====
//  Queues: one set of KVM_NBUCKETS FIFO rings per class (ring index = class * KVM_NBUCKETS + priority). In GLOBAL mode every
//  task goes to class 0's rings, which is exactly the original single queue.
//  Pop (a warp of an idle worker CTA) chooses a class from counters, scans only that class for its best head, claims it:
//    tier 0  class is INHERITED-boosted (another class is parked on a version this class publishes, see below)
//    tier 1  class is below its reservation (running < min_cta): ranked by running/min_cta, so the emptiest reservation first
//    tier 2  the class's best head is within urgent_us of its deadline: earliest deadline first (EDF among the bucket heads;
//            FIFO inside one bucket, so give deadline tasks of different urgency different priorities if exact EDF matters)
//    tier 3  everything else, by virtual time: min over classes of  sum(run_ns) / weight  (start-time fair queueing, charged
//            at claim with the op's running-time estimate and corrected when the task ends, so N idle workers do not all pick
//            the same class). A class that becomes active has its vtime raised to the minimum active vtime: it neither
//            banks credit while idle (work conservation lends idle time out for free) nor is it starved by the others' debt.
//  Task affinity: a ready task is offered only to the block that ran its releasing predecessor for affinity_us (warm L1/code),
//  then to anyone.
//  Caps (max_cta, bw_cta, the arbiter's global BW limit) apply to tiers 2 and 3 only: a reservation or an inheritance boost
//  overrides a throttle, otherwise a throttled class holding the only publisher of a version a reservation-holder waits on
//  would deadlock the waiter. A cap is enforced with fetch_add-then-check-then-rollback, so it is a hard bound.
//  Inside a class the original rule holds: score = priority * aging + age, FIFO per bucket.
//
// ===== Ordering arguments =====
//  * Exactly once is untouched: a task is popped by the single winner of the head CAS (the counters around it are accounting,
//    rolled back on a lost CAS) and enqueued by the single winner of indegree/park CAS as before.
//  * `ready[c]` is a hint that only skips empty classes. It is incremented BEFORE the cell is published and decremented AFTER
//    the head CAS, so it is >= the number of published cells: a stale value costs an empty scan, never a missed task for long
//    (the worker polls again). `running[c]`/`bw_running` are incremented before the head CAS and rolled back when it loses.
//  * Policy parameters are written by the host under a seqlock into mapped memory; only the policy block (block 1) reads them
//    (gen_b acquire, fields relaxed, gen_a recheck) and mirrors them to device memory with plain stores. Workers read the mirror
//    with volatile loads: a stale value delays a policy change by one tick, which is all the guarantee any of this gives.
//
// ===== Priority inheritance across version waits =====
//  A task parked on version variable v (wait_var) is counted in pv[v][its class]; pubcls[v] has the bit of every class that
//  admitted a task publishing v. The policy block (every tick, ~2 us) computes wants[W] = classes W has parked waiters on, and
//  boost = transitive closure of wants (W waits on C, C's own parked task waits on D => C and D boosted). A boosted class is
//  scheduled as if under reservation and ignores every cap, so the publisher (and, class-granular, everything of its class that
//  it needs) is not held back behind a throttle or a low weight. Granularity is the class: unrelated tasks of C ride along.
//  Without KVM_SF_INHERIT a fully throttled publisher class never runs and the waiter parks forever (tested: device TIMEOUT).
//
// ===== Limits, stated plainly =====
//  Tasks are not preemptible: a reclaim (a class below its reservation taking a CTA back) happens at the next task boundary of
//  some CTA, so its worst case is the longest task of the other classes; `resv_wait_max_ns` measures it. Split long tasks
//  into chunks (independent tasks over sub-views of the buffers, see tests/gpu/test_sched.cu) to shorten it.
#include "kvm_sched.h"

// ---------------- operator cost metadata (kvm_ops_cost.inc) ----------------
struct KvmCost { int res; unsigned long long bytes, flops; };
__host__ __device__ constexpr uint32_t kvm_fnv(const char* s) { uint32_t h = 2166136261u; while (*s) { h = (h ^ (uint8_t)*s++) * 16777619u; } return h; }
template <uint32_t H> struct KvmCostOf {  // KVM_COST_DEFAULT: unknown operator
  __host__ __device__ static KvmCost get(const int32_t*) { return {KVM_RES_MIXED, 0ull, 0ull}; }
};
#define U(k) ((unsigned long long)(unsigned)a[k])
#define KC(name, res, bytes, flops) \
  template <> struct KvmCostOf<kvm_fnv(#name)> { __host__ __device__ static KvmCost get(const int32_t* a) { (void)a; return {KVM_RES_##res, (unsigned long long)(bytes), (unsigned long long)(flops)}; } };
#include "kvm_ops_cost.inc"
#undef KC
static const char* const kKvmCostNames[] = {
#define KC(name, ...) #name,
#include "kvm_ops_cost.inc"
#undef KC
};
#undef U
// Cost of one task of op `op` with argument words `a`; a BW task too small to stream is latency-bound.
__host__ __device__ inline KvmCost kvm_cost(int op, const int32_t* a) {
  KvmCost c{KVM_RES_MIXED, 0ull, 0ull};
  switch (op) {
#define X(n, id, sig) case id: c = KvmCostOf<kvm_fnv(#n)>::get(a); break;
#include "kvm_ops_list.inc"
#undef X
    default: break;
  }
  if (c.res == KVM_RES_BW && c.bytes < KVM_LAT_BYTES) c.res = KVM_RES_LAT;
  return c;
}
// Host check: every name in the cost table is a registered operator (a typo would silently fall back to the default).
static inline bool kvm_cost_names_registered() {
  for (const char* n : kKvmCostNames) { bool f = false; for (const KvmOpInfo& o : kKvmOps) f = f || !strcmp(o.name, n); if (!f) return false; }
  return true;
}

#ifdef __CUDACC__
#define KVM_URG      (1ll << 58)     // urgent scores sit above every priority*aging score (31 * 2^50 + age < 2^56)
#define KVM_SLACKMAX (1ll << 54)

// Posted (non-returning) add for accounting counters: the issuing thread does not wait for the L2 round trip.
__device__ __forceinline__ void red_add(int* p, int v) { asm volatile("red.relaxed.gpu.global.add.s32 [%0],%1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ int kvm_vol(const int* p) { return *(const volatile int*)p; }
__device__ __forceinline__ unsigned long long kvm_voll(const unsigned long long* p) { return *(const volatile unsigned long long*)p; }
__device__ __forceinline__ bool kvm_should_exit(const KvmDevState& S, KvmWatch& t0);

// ---------------- enqueue / ready accounting ----------------
// Called by kvm_enqueue before the cell is published (ready is a hint that is never below the number of published cells).
// `n` cells of class c are about to be published (n > 1: the warp-aggregated admission path).
__device__ __forceinline__ void kvm_sched_ready(const KvmDevState& S, int c, int n = 1) {
  KvmSchedDev* D = S.sp.dev; KvmClsDev& k = D->cls[c];
  int old = atom_add(&k.ready, n);
  if (S.sp.mode == KVM_SCHED_DYNAMIC && old <= 0 && kvm_vol(&k.running) == 0) {
    unsigned long long vmin = ~0ull;
    for (int j = 0; j < S.sp.nc; ++j)
      if (j != c && (kvm_vol(&D->cls[j].ready) > 0 || kvm_vol(&D->cls[j].running) > 0)) vmin = min(vmin, kvm_voll(&D->cls[j].vtime));
    if (vmin != ~0ull) atomicMax(&k.vtime, vmin);
  }
}

// The ring index (class, bucket) a task is enqueued on.
__device__ __forceinline__ size_t kvm_queue_of(const KvmDevState& S, const KvmTaskDesc* d) {
  const int c = (S.sp.on && S.sp.mode != KVM_SCHED_GLOBAL) ? d->workload_id : 0;
  return (size_t)c * KVM_NBUCKETS + d->priority;
}

// Fill and publish the cell of ticket `pos` on queue qi (the caller took the ticket and, with the scheduler on, already added the
// task to its class's ready count and, if it has a deadline, to dl_ready).
__device__ __forceinline__ void kvm_publish_cell(const KvmDevState& S, int slot, size_t qi, unsigned long long pos) {
  if (S.fifo_only) {   // one store: the consumer of this ticket sees the slot together with its sequence (no free-cell wait: tasks in flight are far below rcap, and a consumer that ever finds a later generation reports QUEUE_OVERFLOW)
    if (S.trace_on) S.trace[slot].ready_ns = kvm_now();
    st_rel64(&S.cells[pos & (unsigned long long)(S.rcap - 1)].seq, ((pos + 1) << KVM_FIFO_SHIFT) | (unsigned long long)(unsigned)slot);
    return;
  }
  const KvmTaskDesc* d = S.tasks + slot;
  const int c = (int)(qi / KVM_NBUCKETS);
  KvmCell* cell = S.cells + qi * S.rcap + (pos & (unsigned long long)(S.rcap - 1));
  // The cell frees when its previous occupant was popped; the occupant's popper recycles it right after its head CAS.
  for (int spin = 0; ld_acq64(&cell->seq) != pos; ++spin) {
    if (spin > (1 << 22)) { kvm_fail(S, KVM_ERR_QUEUE_OVERFLOW, slot, d->priority, (int)pos); return; }
    __nanosleep(32);
  }
  unsigned long long now = kvm_now();
  S.trace[slot].ready_ns = now;
  cell->slot = slot;
  cell->ready_ns = now;
  cell->deadline_ns = d->deadline_us >= 0 ? S.trace[slot].admit_ns + (unsigned long long)d->deadline_us * 1000ull : 0ull;
  if (S.sp.on) {
    int pref = S.sp.info[slot].pref;   // the block that released the task (chain affinity) ...
    if ((kvm_vol(&S.sp.dev->cls[c].protect) & KVM_CF_OPAFF) && d->op_id < KVM_OP_SLOTS) { int ob = kvm_vol(&S.sp.dev->op_blk[d->op_id]); if (ob >= 0) pref = ob; }   // ... or the one that last ran this op
    cell->res = S.sp.info[slot].res | ((pref + 1) << 8);
  } else cell->res = KVM_RES_MIXED;
  st_rel64(&cell->seq, pos + 1);
  red_add64(&S.qctl[qi].avail, 1ull);   // posted, after the publish: a reservation in kvm_claim that sees it finds a published cell or waits a few cycles
}

__device__ __forceinline__ void kvm_enqueue(const KvmDevState& S, int slot) {
  if (S.fifo_only) { kvm_publish_cell(S, slot, 0, atom_add64(&S.qctl[0].tail, 1ull)); return; }   // no descriptor load: one queue
  const KvmTaskDesc* d = S.tasks + slot;
  const size_t qi = kvm_queue_of(S, d);
  const int c = (int)(qi / KVM_NBUCKETS);
  if (S.sp.on) { kvm_sched_ready(S, c); if (d->deadline_us >= 0) atom_add(&S.sp.dev->cls[c].dl_ready, 1); }
  kvm_publish_cell(S, slot, qi, atom_add64(&S.qctl[qi].tail, 1ull));
}

// ---------------- pop ----------------
struct KvmCand { long long score; int lane; unsigned long long head, dl; int res; };

// Warp-wide scan of class c: the best bucket head by (priority * aging + age), or an urgent EDF score for a head whose deadline
// is within urgent_ns. Heads of BW tasks are skipped when bw_blocked (so a throttled BW head does not block a latency task
// of another bucket). Returns the same value in all lanes.
__device__ __forceinline__ KvmCand kvm_scan(const KvmDevState& S, int c, unsigned long long now, bool bw_blocked, long long urgent_ns, long long aff_ns = 0) {
  const int lane = threadIdx.x & 31;
  const size_t qi = (size_t)c * KVM_NBUCKETS + lane;
  KvmQCtl* q = S.qctl + qi;
  unsigned long long h = kvm_voll(&q->head);   // a hint (the claim CAS validates it); the acquire is the cell's seq below
  KvmCell* cell = S.cells + qi * S.rcap + (h & (unsigned long long)(S.rcap - 1));
  long long score = -1; unsigned long long dl = 0; int res = 0;
  if (ld_acq64(&cell->seq) == h + 1) {
    dl = cell->deadline_ns; res = cell->res & 0xff;
    const int pref = (cell->res >> 8) - 1;   // task affinity: reserved for the block that ran the predecessor for aff_ns
    if (!(bw_blocked && res == KVM_RES_BW) && !(aff_ns && pref >= 0 && pref != (int)blockIdx.x && (long long)(now - cell->ready_ns) < aff_ns)) {
      long long age = (long long)(now - cell->ready_ns);
      score = (long long)lane * S.aging_ns + (age > 0 ? age : 0);
      if (dl && urgent_ns > 0) {
        long long slack = (long long)(dl - now);
        if (slack <= urgent_ns) { slack = max(-KVM_SLACKMAX, min(KVM_SLACKMAX, slack)); score = KVM_URG + (KVM_SLACKMAX - slack); }
      }
    }
  }
  long long best = score; int bl = lane;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    long long o = __shfl_xor_sync(0xffffffffu, best, off);
    int ol = __shfl_xor_sync(0xffffffffu, bl, off);
    if (o > best || (o == best && ol > bl)) { best = o; bl = ol; }
  }
  KvmCand r;
  r.score = best; r.lane = bl;
  r.head = __shfl_sync(0xffffffffu, h, bl); r.dl = __shfl_sync(0xffffffffu, dl, bl); r.res = __shfl_sync(0xffffffffu, res, bl);
  return r;
}

// Claim from the scanned bucket with a reservation and a ticket (the two fetch-adds of perf/admission-bottleneck): a head CAS fails for every
// worker that raced for the same cell (one pop per memory round trip whatever the task, ~0.7 M tasks/s at 107 pollers). Reserve an item with
// a fetch-add on avail (published minus reserved), then take the bucket's next ticket with a fetch-add on head: no retries, FIFO per bucket;
// the ticket's cell is the next published item, or will be within a few instructions (its enqueuer is between its tail ticket and the
// release store). After the claim the lanes fetch what the worker needs (the descriptor, one word per lane < KVM_TASK_WORDS, and the admission
// estimate) in one round trip. Returns the slot in every lane, or -2 when the bucket was emptied first.
__device__ __forceinline__ int kvm_claim(const KvmDevState& S, int c, const KvmCand& cd, int& dw, unsigned& est, bool want_est) {
  const int lane = threadIdx.x & 31;
  const size_t qi = (size_t)c * KVM_NBUCKETS + cd.lane;
  KvmQCtl* q = S.qctl + qi;
  int got = -2;
  if (lane == cd.lane && (long long)ld_acq64(&q->avail) > 0) {          // look before reserving: a failed reservation is a decrement plus a restore on the contended line
    if ((long long)atom_add64(&q->avail, ~0ull) > 0) {
      const unsigned long long h = atom_add64(&q->head, 1ull);
      KvmCell* cell = S.cells + qi * S.rcap + (h & (unsigned long long)(S.rcap - 1));
      int spin = 0;
      while (ld_acq64(&cell->seq) != h + 1) {                            // its enqueuer publishes within a few instructions
        if (++spin > (1 << 22) || *(volatile int*)&S.cnt->err_code) { kvm_fail(S, KVM_ERR_QUEUE_OVERFLOW, -1, 99, (int)h); break; }
        __nanosleep(16);
      }
      if (spin <= (1 << 22) && !*(volatile int*)&S.cnt->err_code) {
        got = cell->slot;
        st_rel64(&cell->seq, h + (unsigned long long)S.rcap);           // recycle before running
      }
    } else atom_add64(&q->avail, 1ull);                                 // lost the race for the last item: give the reservation back
  }
  got = __shfl_sync(0xffffffffu, got, cd.lane);
  const bool valid = got >= 0 && got < S.nslots;
  dw = (lane < KVM_TASK_WORDS && valid) ? __ldcg((const int*)(S.tasks + got) + lane) : 0;
  est = (want_est && valid) ? __ldcg(&S.sp.info[got].est_ns) : 0;
  return got;
}

// FIFO mode (every task priority 0, no deadlines, scheduler off): lane 0 takes a ticket on the one ready ring when it has none, then watches the cell of that ticket (no scan of 32
// buckets, no reservation, no CAS retries). Every lane returns the slot, or KVM_POP_EXIT when the run has ended. A ticket survives until its cell fills; a block leaves holding one only
// at the end of the run (kvm_should_exit: every admitted task is done and admission is closed), when no further task can land in its cell.
__device__ __forceinline__ int kvm_fifo_pop(const KvmDevState& S, KvmWatch& t0, long long& ticket, int& dw) {
  const int lane = threadIdx.x & 31;
  int got = -1;
  if (lane == 0) {
    if (ticket < 0) ticket = (long long)atom_add64(&S.qctl[0].head, 1ull);
    const unsigned long long want = (unsigned long long)ticket + 1;
    KvmCell* cell = S.cells + ((unsigned long long)ticket & (unsigned long long)(S.rcap - 1));
    for (int spin = 0;; ++spin) {
      const unsigned long long v = ld_acq64(&cell->seq), hi = v >> KVM_FIFO_SHIFT;
      if (hi == want) { got = (int)(v & ((1ull << KVM_FIFO_SHIFT) - 1)); ticket = -1; break; }
      if (hi > want) { kvm_fail(S, KVM_ERR_QUEUE_OVERFLOW, -1, 98, (int)ticket); got = KVM_POP_EXIT; break; }
      if ((spin & 15) == 15) {
        if (kvm_should_exit(S, t0)) { got = KVM_POP_EXIT; break; }
        if (spin > 64) __nanosleep(64);
      }
    }
  }
  got = __shfl_sync(0xffffffffu, got, 0);
  dw = (lane < KVM_TASK_WORDS && got >= 0) ? __ldcg((const int*)(S.tasks + got) + lane) : 0;
  return got;
}

// STATIC mode: the class that owns this worker's SM.
__device__ __forceinline__ int kvm_sched_part(const KvmDevState& S) {
  if (!S.sp.on || S.sp.mode != KVM_SCHED_STATIC) return -1;
  const int sm = kvm_smid();
  for (int c = 0; c < S.sp.nc - 1; ++c) if (sm < kvm_vol(&S.sp.dev->cls[c].sm_hi)) return c;
  return S.sp.nc - 1;
}

// Warp-wide (all 32 lanes call). Returns a slot or -1; `how` = KVM_TRACE_* start-reason bits (same in all lanes); lane l < KVM_TASK_WORDS
//  returns word l of the claimed task's descriptor in `dw` (prefetched during the claim).
//  GLOBAL/STATIC serve exactly one class (0 or the worker's SM-owner), so they scan it and claim. DYNAMIC first decides the
//  class from counters alone (two coalesced loads per class, all in flight at once), and scans only the chosen class; a class
//  is scanned before the decision only when it holds ready deadline tasks and has a deadline window (urgency needs the head).
__device__ __forceinline__ int kvm_pop(const KvmDevState& S, unsigned long long now, int part, int& how, int& dw) {
  const int lane = threadIdx.x & 31;
  const KvmSchedPtrs& sp = S.sp;
  how = 0; dw = 0;
  if (!sp.on) {  // the original executor: one class, no accounting
    for (int attempt = 0; attempt < 4; ++attempt) {
      KvmCand cd = kvm_scan(S, 0, now, false, 0);
      if (cd.score < 0) return -1;
      unsigned est;
      int got = kvm_claim(S, 0, cd, dw, est, false);
      if (got >= 0) return got;
    }
    return -1;
  }
  KvmSchedDev* D = sp.dev;
  if (sp.mode != KVM_SCHED_DYNAMIC) {
    const int c = sp.mode == KVM_SCHED_STATIC ? part : 0;
    KvmClsDev& k = D->cls[c];
    const long long aff = kvm_vol(&D->affinity_ns);
    const bool acct = sp.mode == KVM_SCHED_STATIC;   // GLOBAL counts a class's running tasks at task start (the class is only known then)
    for (int attempt = 0; attempt < 4; ++attempt) {
      if (__shfl_sync(0xffffffffu, kvm_vol(&k.ready), 0) <= 0) return -1;
      KvmCand cd = kvm_scan(S, c, now, false, 0, aff);
      if (cd.score < 0) return -1;
      const bool isbw = cd.res == KVM_RES_BW;
      if (acct && lane == 0) { red_add(&k.running, 1); if (isbw) { red_add(&k.bw_running, 1); red_add(&D->bw_total, 1); } }
      unsigned est;
      int got = kvm_claim(S, c, cd, dw, est, false);
      if (got < 0) { if (acct && lane == 0) { red_add(&k.running, -1); if (isbw) { red_add(&k.bw_running, -1); red_add(&D->bw_total, -1); } } continue; }
      if (lane == 0) { red_add(&k.ready, -1); if (cd.dl) red_add(&k.dl_ready, -1); }
      if (isbw) how |= KVM_HOW_BW;
      return got;
    }
    return -1;
  }
  const int nc = sp.nc;
  const int flags = kvm_vol(&D->flags);
  const int boost = (flags & KVM_SF_INHERIT) ? kvm_vol(&D->boost_mask) : 0;
  const bool arb = (flags & KVM_SF_ARBITER) != 0;
  const long long aff = kvm_vol(&D->affinity_ns);
  const int bw_total = arb ? kvm_vol(&D->bw_total) : 0, bw_limit = arb ? kvm_vol(&D->bw_limit) : 0;
  unsigned excl = 0, bwblk = 0;
  for (int attempt = 0; attempt < 6; ++attempt) {
    int pw[KVM_MAX_CLS], hw[KVM_MAX_CLS];   // lane i holds word i of the class's policy line / hot line
#pragma unroll
    for (int c = 0; c < KVM_MAX_CLS; ++c) if (c < nc) { pw[c] = __ldcg((const int*)&D->cls[c] + lane); hw[c] = __ldcg((const int*)&D->cls[c].running + lane); }
    unsigned long long bkey = ~0ull; int bc = -1, btier = 3, bmx = 0, bbwc = 0; bool bhave = false, bbw = false; KvmCand bcd{};
#pragma unroll
    for (int c = 0; c < KVM_MAX_CLS; ++c) {
      if (c >= nc || ((excl >> c) & 1)) continue;
      const int ready = __shfl_sync(0xffffffffu, hw[c], 2);
      if (ready <= 0) continue;
      const int run = __shfl_sync(0xffffffffu, hw[c], 0), bwr = __shfl_sync(0xffffffffu, hw[c], 1), dlr = __shfl_sync(0xffffffffu, hw[c], 3);
      const unsigned long long vt = (unsigned long long)(unsigned)__shfl_sync(0xffffffffu, hw[c], 4) | ((unsigned long long)__shfl_sync(0xffffffffu, hw[c], 5) << 32);
      const int mn = (flags & KVM_SF_GUARANTEE) ? __shfl_sync(0xffffffffu, pw[c], 1) : 0, mx = __shfl_sync(0xffffffffu, pw[c], 2);
      const int bwc = __shfl_sync(0xffffffffu, pw[c], 3), urg_us = __shfl_sync(0xffffffffu, pw[c], 4);
      int tier = ((boost >> c) & 1) ? 0 : run < mn ? 1 : 3;
      bool bwb = (bwblk >> c) & 1; bool have = false; KvmCand cd{};
      if (tier >= 2) {  // throttles apply only to unprivileged service
        if (run >= mx) continue;
        bwb = bwb || bwr >= bwc || (arb && bw_total >= bw_limit);
        if (urg_us > 0 && dlr > 0) {
          cd = kvm_scan(S, c, now, bwb, (long long)urg_us * 1000, aff); have = true;
          if (cd.score < 0) continue;
          if (cd.score >= KVM_URG) tier = 2;
        }
      }
      unsigned long long sub;
      if (tier == 1) sub = ((unsigned long long)run << 20) / (unsigned)max(mn, 1);
      else if (tier == 2) sub = (unsigned long long)(max(-KVM_SLACKMAX, min(KVM_SLACKMAX, (long long)(cd.dl - now))) + KVM_SLACKMAX);
      else sub = vt;
      unsigned long long key = ((unsigned long long)tier << 60) | min(sub, (1ull << 60) - 1);
      if (key < bkey) { bkey = key; bc = c; btier = tier; bcd = cd; bhave = have; bbw = bwb; bmx = mx; bbwc = bwc; }
    }
    if (bc < 0) return -1;
    if (!bhave) { bcd = kvm_scan(S, bc, now, bbw, 0, aff); if (bcd.score < 0) { excl |= 1u << bc; continue; } }
    // A class with a finite cap is counted BEFORE the head CAS (fetch_add, check, roll back: a hard bound). A class without one
    // is counted after the CAS with posted adds, which keeps the hot line off the critical path of a 300-CTA herd.
    int ok = 1, why = 0;
    const bool isbw = bcd.res == KVM_RES_BW;
    const bool pre_run = btier >= 2 && bmx != KVM_UNLIMITED, pre_bw = isbw && btier >= 2 && (bbwc != KVM_UNLIMITED || arb), pre = pre_run || pre_bw;
    KvmClsDev& bk = D->cls[bc];
    if (pre && lane == 0) {
      int old = atom_add(&bk.running, 1);
      if (btier >= 2 && old >= bmx) { red_add(&bk.running, -1); ok = 0; why = 1; }
      else if (isbw) {
        int ob = atom_add(&bk.bw_running, 1), og = atom_add(&D->bw_total, 1);
        if (btier >= 2 && (ob >= bbwc || (arb && og >= kvm_vol(&D->bw_limit)))) {
          red_add(&bk.bw_running, -1); red_add(&D->bw_total, -1); red_add(&bk.running, -1); ok = 0; why = 2;
        }
      }
    }
    ok = __shfl_sync(0xffffffffu, ok, 0); why = __shfl_sync(0xffffffffu, why, 0);
    if (!ok) { if (why == 1) excl |= 1u << bc; else bwblk |= 1u << bc; continue; }
    unsigned est;
    int got = kvm_claim(S, bc, bcd, dw, est, true);
    if (got < 0) {
      if (pre && lane == 0) { red_add(&bk.running, -1); if (isbw) { red_add(&bk.bw_running, -1); red_add(&D->bw_total, -1); } }
      continue;
    }
    if (!pre && lane == 0) { red_add(&bk.running, 1); if (isbw) { red_add(&bk.bw_running, 1); red_add(&D->bw_total, 1); } }
    if (lane == 0) {
      red_add(&bk.ready, -1);
      if (bcd.dl) red_add(&bk.dl_ready, -1);
      if (est == 0) est = 2000;
      S.sp.info[got].est_ns = (int)est;
      atomicAdd(&bk.vtime, (unsigned long long)est * (unsigned long long)kvm_vol(&bk.vt_mul));
    }
    how = (btier == 0 ? KVM_TRACE_BOOST : 0) | (btier == 1 ? KVM_TRACE_RESV : 0) | (btier == 2 ? KVM_TRACE_URGENT : 0) | (isbw ? KVM_HOW_BW : 0);
    return got;
  }
  return -1;
}

// ---------------- task start / end accounting (thread 0 of the worker) ----------------
__device__ __forceinline__ void kvm_sched_start(const KvmDevState& S, int slot, int cls, int how, unsigned long long start) {
  if (!S.sp.on) return;
  KvmSchedDev* D = S.sp.dev; KvmClsDev& k = D->cls[cls];
  if (S.sp.mode == KVM_SCHED_GLOBAL) red_add(&k.running, 1);  // other modes counted it at claim
  const unsigned long long w = start - S.trace[slot].ready_ns;
  atomicAdd(&k.w_wait_sum, w); atomicAdd(&k.w_n, 1ull); atomicMax(&k.w_wait_max, w); atomicAdd(&k.wait_sum, w);
  { const int tw = kvm_vol(&k.target_wait_us); if (tw > 0 && w > (unsigned long long)tw * 1000ull) atomicAdd(&k.wait_over, 1ull); }
  if (how & KVM_TRACE_RESV) { atomicAdd(&k.resv_starts, 1ull); atomicMax(&k.resv_wait_max_ns, w); }
}
// Before the next pop of this block (thread 0): the class stops counting this CTA as running. Posted adds only, no loads.
__device__ __forceinline__ void kvm_sched_release(const KvmDevState& S, int cls, bool bw) {
  if (!S.sp.on) return;
  KvmSchedDev* D = S.sp.dev; KvmClsDev& k = D->cls[cls];
  red_add(&k.running, -1);
  if (bw) { red_add(&k.bw_running, -1); red_add(&D->bw_total, -1); }
}
// Deferred (helper thread, during the next pop): statistics and the running-time estimate; must run before the slot is recycled.
__device__ __forceinline__ void kvm_sched_account(const KvmDevState& S, int slot, unsigned long long start, unsigned long long end, bool missed) {
  if (!S.sp.on) return;
  KvmSchedDev* D = S.sp.dev; const KvmTaskDesc& d = S.tasks[slot]; KvmClsDev& k = D->cls[d.workload_id];
  const KvmSlotInfo& in = S.sp.info[slot];
  const unsigned long long run = end - start;
  if (in.res == KVM_RES_BW) atomicAdd(&D->bw_bytes, in.bytes);
  atomicAdd(&k.sm_ns, run); atomicAdd(&k.tasks, 1ull); atomicAdd(&k.bytes, in.bytes);
  if (missed) { atomicAdd(&k.missed, 1ull); atomicAdd(&k.w_miss, 1ull); }
  if (d.deadline_us >= 0) {   // deadline pressure: finishing in the last quarter of the budget is a near miss (the controller acts before a miss, not after)
    const long long budget = (long long)d.deadline_us * 1000ll, left = budget - (long long)(end - S.trace[slot].admit_ns);
    atomicAdd(&k.dl_n, 1ull); if (left < budget / 4) atomicAdd(&k.near_miss, 1ull);
  }
  if (d.op_id < KVM_OP_SLOTS) D->op_blk[d.op_id] = (int)blockIdx.x;
  if (d.op_id < KVM_OP_SLOTS) {  // racy by design: an estimate
    unsigned old = D->ema_ns[d.op_id];
    D->ema_ns[d.op_id] = old ? (unsigned)((old * 7ull + min(run, 0xffffffffull)) / 8) : (unsigned)min(run, 0xffffffffull);
  }
  if (S.sp.mode == KVM_SCHED_DYNAMIC && in.est_ns)  // correct the claim-time charge by the actual running time
    atomicAdd(&k.vtime, (unsigned long long)(((long long)run - (long long)in.est_ns) * (long long)kvm_vol(&k.vt_mul)));
}

// The releasing block becomes the task's preferred block (see KvmSchedParams.affinity_us). The write is ordered before the
// enqueue's release store of the cell, which is what the scanner's acquire load pairs with.
__device__ __forceinline__ void kvm_sched_pref(const KvmDevState& S, int slot) { if (S.sp.on) S.sp.info[slot].pref = (int)blockIdx.x; }

// ---------------- version-wait accounting (priority inheritance input) ----------------
__device__ __forceinline__ void kvm_sched_park(const KvmDevState& S, int slot) {
  if (S.sp.on && S.tasks[slot].workload_id >= 0) atom_add(S.sp.pv + (size_t)S.tasks[slot].wait_var * KVM_MAX_CLS + S.tasks[slot].workload_id, 1);
}
__device__ __forceinline__ void kvm_sched_unpark(const KvmDevState& S, int slot) {
  if (S.sp.on) atom_add(S.sp.pv + (size_t)S.tasks[slot].wait_var * KVM_MAX_CLS + S.tasks[slot].workload_id, -1);
}

// ---------------- admission: class check, publisher classes, cost ----------------
__device__ __forceinline__ bool kvm_sched_class_ok(const KvmDevState& S, int wid) { return !S.sp.on || (wid >= 0 && wid < S.sp.nc); }
__device__ __forceinline__ void kvm_sched_admit_task(const KvmDevState& S, int slot) {  // args already copied
  if (!S.sp.on) return;
  const KvmTaskDesc& d = S.tasks[slot];
  KvmCost c = kvm_cost(d.op_id, S.args + d.args_off);
  int est = kvm_vol((const int*)&S.sp.dev->ema_ns[min(d.op_id, KVM_OP_SLOTS - 1)]);   // the op's running-time estimate now: the pop charges it without a load
  S.sp.info[slot] = KvmSlotInfo{c.bytes, c.res, est < 0 ? 0x7fffffff : est, -1, 0};
  if (d.pub_var >= 0) atomicOr(S.sp.pubcls + d.pub_var, 1 << d.workload_id);
}

// ---------------- policy block (block 1) ----------------
struct KvmPolShared {
  int exit, g1, ok, last_gen, do_export, wants[KVM_MAX_CLS];
  int window_us, lim_lo, lim_hi;                 // arbiter parameters of the last consistent parameter set
  unsigned long long ticks;
  int words[sizeof(KvmSchedParams) / 4];
  int win_avg_us[KVM_MAX_CLS], win_max_us[KVM_MAX_CLS], win_n[KVM_MAX_CLS];
  int bw_mbps, bw_peak_mbps;
};

// One block, all threads. Every tick (~2 us): pick up new host parameters, recompute the inheritance boost, every window_us
// measure achieved bandwidth and run the arbiter (AIMD on the global concurrent-BW-task limit), every ~20 us export telemetry.
__device__ void kvm_policy(const KvmDevState& S, KvmWatch& t0, KvmPolShared& P) {
  const KvmSchedPtrs& sp = S.sp;
  KvmSchedDev* D = sp.dev;
  const int tid = threadIdx.x, nc = sp.nc;
  unsigned long long win_start = kvm_now(), last_bytes = 0, peak = 0;
  int calm = 0, cool = 0;
  if (tid == 0) { P.last_gen = -1; P.ticks = 0; P.window_us = 200; P.lim_lo = 8; P.lim_hi = 1 << 20; for (int c = 0; c < KVM_MAX_CLS; ++c) P.win_avg_us[c] = P.win_max_us[c] = P.win_n[c] = 0; P.bw_mbps = P.bw_peak_mbps = 0; }
  __syncthreads();
  for (;;) {
    if (tid == 0) { P.exit = kvm_should_exit(S, t0); P.g1 = (int)ld_acq_sys((const unsigned*)&sp.hp->gen_b); P.do_export = (P.ticks++ % 8) == 0; }
    __syncthreads();
    if (P.exit) return;
    // 1. parameters: seqlock read (gen_b acquire, fields, gen_a); apply when consistent and new
    if (P.g1 != P.last_gen) {
      for (int w = tid; w < (int)(sizeof(KvmSchedParams) / 4); w += blockDim.x) P.words[w] = ld_sys(((const int*)sp.hp) + w);
      __syncthreads();
      if (tid == 0) { P.ok = ld_sys((const int*)&sp.hp->gen_a) == P.g1; if (P.ok) P.last_gen = P.g1; }
      __syncthreads();
      if (P.ok) {
        const KvmSchedParams* q = (const KvmSchedParams*)P.words;
        if (tid == 0) {
          D->flags = q->flags; D->affinity_ns = q->affinity_us * 1000;
          P.window_us = max(q->window_us, 20); P.lim_lo = q->bw_limit_min; P.lim_hi = q->bw_limit_max;
          D->bw_limit = max(P.lim_lo, min(P.lim_hi, kvm_vol(&D->bw_limit)));
        }
        if (tid < nc) {
          KvmClsDev& k = D->cls[tid]; const KvmClassParams& p = q->cls[tid];
          k.weight = p.weight; k.min_cta = p.min_cta; k.max_cta = p.max_cta; k.bw_cta = p.bw_cta; k.urgent_us = p.urgent_us;
          k.protect = p.protect; k.target_wait_us = p.target_wait_us; k.sm_hi = p.sm_hi; k.vt_mul = (1 << 20) / max(p.weight, 1);
        }
      }
      __syncthreads();
    }
    const int flags = kvm_vol(&D->flags);
    // 2. inheritance boost
    if (tid < KVM_MAX_CLS) P.wants[tid] = 0;
    __syncthreads();
    if (flags & KVM_SF_INHERIT)
      for (int v = tid; v < S.num_vars; v += blockDim.x) {
        int pc = kvm_vol(&sp.pubcls[v]);
        if (pc) for (int w = 0; w < nc; ++w) if (kvm_vol(&sp.pv[(size_t)v * KVM_MAX_CLS + w]) > 0) atomicOr(&P.wants[w], pc & ~(1 << w));
      }
    __syncthreads();
    if (tid == 0) {
      int B = 0;
      for (int w = 0; w < nc; ++w) B |= P.wants[w];
      for (int it = 0; it < nc; ++it) for (int c = 0; c < nc; ++c) if ((B >> c) & 1) B |= P.wants[c];
      D->boost_mask = B;
      // 3. bandwidth arbiter window
      unsigned long long now = kvm_now();
      if (now - win_start >= (unsigned long long)P.window_us * 1000ull) {
        unsigned long long bytes = kvm_voll(&D->bw_bytes), dt = now - win_start;
        unsigned long long rate = (bytes - last_bytes) * 1000ull / dt;   // MB/s (bytes per ns * 1000)
        peak = max(peak - (peak >> 8), rate);
        bool pressure = false;
        for (int c = 0; c < nc; ++c) {
          KvmClsDev& k = D->cls[c];
          unsigned long long wn = atomicExch(&k.w_n, 0ull), ws = atomicExch(&k.w_wait_sum, 0ull), wm = atomicExch(&k.w_wait_max, 0ull), mi = atomicExch(&k.w_miss, 0ull);
          P.win_n[c] = (int)wn; P.win_avg_us[c] = wn ? (int)(ws / wn / 1000) : 0; P.win_max_us[c] = (int)(wm / 1000);
          int tgt = kvm_vol(&k.target_wait_us);
          (void)mi;   // deadline misses are reported, not used: an unattainable SLO would pin the limit at its floor
          if ((kvm_vol(&k.protect) & KVM_CF_PROTECT) && wn && tgt > 0 && (int)(wm / 1000) > tgt) pressure = true;
        }
        if (flags & KVM_SF_ARBITER) {
          int lim = kvm_vol(&D->bw_limit), lo = P.lim_lo, hi = P.lim_hi;
          if (cool > 0) --cool;
#ifndef KVM_ARB_DEC
#define KVM_ARB_DEC 4     // multiplicative decrease: lim -= lim / KVM_ARB_DEC
#endif
#ifndef KVM_ARB_CALM
#define KVM_ARB_CALM 4    // calm windows before an increase
#endif
#ifndef KVM_ARB_INC
#define KVM_ARB_INC 16    // increase: lim += 1 + lim / KVM_ARB_INC
#endif
          if (pressure) { lim = max(lo, lim - max(1, lim / KVM_ARB_DEC)); calm = 0; cool = 2; }          // multiplicative decrease
          else if (++calm >= KVM_ARB_CALM && cool == 0 && kvm_vol(&D->bw_total) * 10 >= lim * 9) { lim = min(hi, lim + 1 + lim / KVM_ARB_INC); calm = 0; }  // slow increase when binding
          D->bw_limit = lim;
        }
        P.bw_mbps = (int)min(rate, 0x7fffffffull); P.bw_peak_mbps = (int)min(peak, 0x7fffffffull);
        last_bytes = bytes; win_start = now;
      }
    }
    __syncthreads();
    // 4. telemetry export to the mapped block
    if (P.do_export) {
      if (tid < nc) {
        KvmClsDev& k = D->cls[tid]; KvmClassTele* t = &sp.ht->cls[tid];
        t->sm_ns = kvm_voll(&k.sm_ns); t->done = kvm_voll(&k.tasks); t->bytes = kvm_voll(&k.bytes); t->missed = kvm_voll(&k.missed);
        t->dl_n = kvm_voll(&k.dl_n); t->near_miss = kvm_voll(&k.near_miss); t->wait_over = kvm_voll(&k.wait_over); t->wait_sum_ns = kvm_voll(&k.wait_sum);
        t->running = kvm_vol(&k.running); t->ready = kvm_vol(&k.ready);
        t->win_wait_avg_us = P.win_avg_us[tid]; t->win_wait_max_us = P.win_max_us[tid]; t->win_n = P.win_n[tid];
        t->resv_max_us = (int)min(kvm_voll(&k.resv_wait_max_ns) / 1000ull, 0x7fffffffull);
      } else if (tid == 32) {
        KvmSchedTele* t = sp.ht;
        t->bw_mbps = P.bw_mbps; t->bw_peak_mbps = P.bw_peak_mbps; t->bw_limit = kvm_vol(&D->bw_limit); t->boost_mask = kvm_vol(&D->boost_mask);
        t->params_gen = P.last_gen; t->bw_running = kvm_vol(&D->bw_total); t->ticks = P.ticks;
      }
    }
    if (tid < 32) __nanosleep(1000);
    __syncthreads();
  }
}
#endif  // __CUDACC__
