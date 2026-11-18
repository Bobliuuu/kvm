#pragma once
// Magni VM static schedules, device side: one persistent kernel whose blocks each run a program compiled before launch (mvm_static.h).
//
//   * A worker (one block of MVM_THREADS threads) executes its records in order. Records are fully resolved: op, argument words and buffer views
//     are read from shared memory, where the next record is copied (cp.async) while the current one runs.
//   * Between two records of one worker the only synchronization is __syncthreads(). A dependency across workers is a wait on the
//     publishing worker's watermark: thread 0 polls progress[w] with an acquire load, then a barrier hands the data to the block.
//   * A record that another worker waits for publishes: every warp fences (fence.sc.gpu makes ITS stores visible across SMs, mvm_exec.cuh's
//     rule), a barrier, thread 0's release store of the watermark. Nothing else costs an atomic or a queue.
// Memory-ordering, with the waiter's thread 0 and the publisher's thread 0 as the anchors: the waiter's ld.acquire.gpu synchronizes with the
// publisher's st.release.gpu, whose fence-then-barrier history covers every store of every record the publisher ran before it; the
// waiter's barrier extends that to all of its threads (mvm_exec.cuh's argument, unchanged).
#include "mvm_static.h"
#include "mvm_fuse.h"
#include "mvm_fused_mlp.cuh"
#include "mvm_fused_split.cuh"
#include "mvm_fused_grid.cuh"
#include "mvm_fused_grid_virtual.cuh"
#include "mvm_fused_batch.cuh"

// The dynamic shared-memory arena every op takes from offset 0 (the same declaration as mvm_ops_gpt.cuh's, repeated so a build without the gpt pack, like tests/gpu/build_core.sh, still has it).
extern __shared__ __align__(128) unsigned char mvm_gpt_smem[];

struct MvmStaticDev {
  const MvmSRec* recs;            // every worker's records, concatenated
  const int* offs;                // [nworkers + 1]: worker w runs recs[offs[w] .. offs[w+1])
  int* progress;                  // [nworkers * MVM_S_LINE_WORDS]: per worker a monotone watermark on its own 128-byte line
  int nworkers, trace_on;
  int nactive;                    // workers with a program (the rest exit at once, or spin with spin_idle)
  int spin_idle;                  // 1: a worker with no program polls `finished` until every active worker is done (keeps the SMs busy for the clock governor)
  int* finished;                  // [1] workers that have completed their program
  unsigned long long timeout_ns;  // stall limit of one wait: a worker that waits this long fails the run (a missing publisher is a bug)
  MvmCounters* cnt;
  MvmHostErr* herr;
  unsigned long long* trace;      // [nrecs * MVM_S_TRACE_WORDS]: start, waits done, op done, published (nullptr when off)
  const MvmFusedDesc* fused;      // the fused runs of this bundle (mvm_fuse.h); nullptr when it has none
  const MvmSplitDesc* split;      // the split fused runs of this bundle (mvm_fuse.h's split form); nullptr when it has none
  const MvmGridDesc* grid;        // the grid fused runs of this bundle (mvm_fuse.h's grid form); nullptr when it has none
  const MvmBatchDesc* batch;      // the batched gangs of this bundle (mvm_fuse.h's batched form); nullptr when it has none
  long long* fuse_phases;         // [nworkers * MVM_FMLP_NPHASES]: per worker, cycles in each phase of its slices (--fuse-phases; nullptr when off)
};
#define MVM_S_LINE_WORDS 32         // one watermark per 128-byte line: one writer, a few pollers, no false sharing
#define MVM_S_TRACE_WORDS 4

#ifdef __CUDACC__
struct MvmSShared { MvmSRec rec[2]; int stop; long long t_phase; float smem[MVM_SMEM_FLOATS]; };   // two record slots: current and being fetched

// Copy one 512-byte record into shared memory: one 16-byte cp.async per lane of warp 0, committed as one group per lane.
__device__ __forceinline__ void mvm_s_fetch(const MvmSRec* g, MvmSRec* s) {
  if (threadIdx.x < 32) {
    const unsigned d = (unsigned)__cvta_generic_to_shared((char*)s + 16 * threadIdx.x);
    const char* src = (const char*)g + 16 * threadIdx.x;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src) : "memory");
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
}
__device__ __forceinline__ void mvm_s_fetch_wait() { if (threadIdx.x < 32) asm volatile("cp.async.wait_all;" ::: "memory"); }

// Thread 0: poll one watermark until it reaches `need`. Returns false on a stall or a failure elsewhere.
__device__ __forceinline__ bool mvm_s_await(const MvmStaticDev& S, int worker, int need, int task) {
  int* p = S.progress + (size_t)worker * MVM_S_LINE_WORDS;
  unsigned long long t0 = 0;
  for (unsigned spin = 0;; ++spin) {
    if (ld_acq(p) >= need) return true;
    if ((spin & 0xfff) == 0xfff) {     // every ~1-2 ms of polling
      if (*(volatile int*)&S.cnt->err_code) return false;
      const unsigned long long n = mvm_now();
      if (!t0) t0 = n;
      else if (n - t0 > S.timeout_ns) { mvm_fail(S.cnt, S.herr, MVM_ERR_TIMEOUT, task, worker, need); return false; }
    }
  }
}

// The fused ops' bodies as functions of their own, not inlined into the worker loop: the loop's live state then stays out of their register allocation, and the GEMMs' accumulators
// do not spill around it (measured: with them inlined the static kernel ran 255 registers and a 3.6 KB stack frame; the same body on the bed, alone, 64 bytes).
__device__ __noinline__ void mvm_s_fmlp_slice(int sc, const int32_t* ids, const int32_t* tgt, float* arena, long long* ph, long long* ph_last, MvmCtx* c) {
  mvm_fmlp::slice_run(sc, ids, tgt, nullptr, arena, mvm_fmlp::PhaseTimer{ph, ph_last}, [c](int code, int v) { c->fail(code, v); });
}
__device__ __noinline__ void mvm_s_fmlp_sgd(int owner, float* loss, float* arena) { mvm_fmlp::sgd_run(owner, loss, arena); }
// One member of a split step at S_ shares (a gang op: the planner placed its members on distinct workers at one point of every program,
// mvm_static_plan). Each virtual share count is a function of its own, on an entry of its own (mvm_static_gang_split<S_>): with all of them in
// one function, the member used every register and still spilled (sm_80, Nov 11). The gang's physical share count is the descriptor's S, any count
// up to its Sv (mvm_fused_split.h). A descriptor whose Sv and Q are not this entry's, an S outside 1..Sv, or a member count that disagrees with
// them, fails the run before any barrier. Every member reads the same descriptor, so all of them fail alike. Any other
// failure ends the gang at its next crossing: the barrier watches the run's error word.
template <int S_>
__device__ __noinline__ void mvm_s_fmlp_split(int member, int members, const int32_t* ids, const int32_t* tgt, float* loss, float* arena, MvmCtx* c) {
  mvm_fmlp::cp_commit_wait();
  __syncthreads();
  const MvmSplitDesc& d = *reinterpret_cast<const MvmSplitDesc*>(arena);
  if (d.Q != MVM_SPLIT_Q || d.Sv != S_ || d.S < 1 || d.S > d.Sv || members != d.Q * d.S || member < 0 || member >= members) { c->fail(MVM_ERR_BAD_OP, MVM_OP_fmlp_split); return; }
  mvm_fsplit::split_run<S_, MVM_SPLIT_Q>(member, ids, tgt, loss, arena, &c->cnt->err_code, mvm_fsplit::NoPhases{});
}
// The same member with its weights streamed (mvm_fused_split.h: the same bits, in a memory that does not grow with the run's weights), on an
// entry of its own so the resident member's registers stay as they are. Besides the resident entry's checks, a descriptor not laid out for
// streaming (mvm_split_is_streamed) fails the run before any barrier: this member would read its staging tiles where the layout put weights.
template <int S_, bool ODX>
__device__ __noinline__ void mvm_s_fmlp_split_stream(int member, int members, const int32_t* ids, const int32_t* tgt, float* loss, float* arena, MvmCtx* c) {
  mvm_fmlp::cp_commit_wait();
  __syncthreads();
  const MvmSplitDesc& d = *reinterpret_cast<const MvmSplitDesc*>(arena);
  if (d.Q != MVM_SPLIT_Q || d.Sv != S_ || d.S < 1 || d.S > d.Sv || members != d.Q * d.S || member < 0 || member >= members || !mvm_split_is_streamed(d)) { c->fail(MVM_ERR_BAD_OP, MVM_OP_fmlp_split); return; }
  mvm_fsplit::split_run<S_, MVM_SPLIT_Q, true, ODX>(member, ids, tgt, loss, arena, &c->cnt->err_code, mvm_fsplit::NoPhases{});
}

// One member of a grid step (a gang op, like fmlp_split). A member count that disagrees with the descriptor fails the run before any barrier;
// any other failure ends the gang at its next crossing, as for the split step.
__device__ __noinline__ void mvm_s_fmlp_grid(int member, int members, const int32_t* ids, const int32_t* tgt, float* loss, float* arena, MvmCtx* c) {
  mvm_fmlp::cp_commit_wait();
  __syncthreads();
  const MvmGridDesc& d = *reinterpret_cast<const MvmGridDesc*>(arena);
  if (members != d.P || member < 0 || member >= members) { c->fail(MVM_ERR_BAD_OP, MVM_OP_fmlp_grid); return; }
  mvm_fgrid::grid_run(member, ids, tgt, loss, arena, &c->cnt->err_code, mvm_fgrid::NoPhases{});
}

// One block of a virtual grid step (mvm_fused_grid_virtual.cuh: grid_run_chunked, each block a contiguous chunk of the gang's roles). A block
// count that disagrees with the descriptor's physical count, or a descriptor not laid out chunked, fails the run before any barrier, as for the
// resident member.
__device__ __noinline__ void mvm_s_fmlp_grid_virtual(int member, int members, const int32_t* ids, const int32_t* tgt, float* loss, float* arena, MvmCtx* c) {
  mvm_fmlp::cp_commit_wait();
  __syncthreads();
  const MvmGridDesc& d = *reinterpret_cast<const MvmGridDesc*>(arena);
  if (d.phys < 1 || d.phys >= d.P || d.G < 1 || members != d.phys || member < 0 || member >= members) { c->fail(MVM_ERR_BAD_OP, MVM_OP_fmlp_grid); return; }
  mvm_fgrid::grid_run_chunked(member, ids, tgt, loss, arena, &c->cnt->err_code, mvm_fgrid::NoPhases{});
}

// One member of a batched step (a gang op of every worker, like fmlp_grid): step `step` of every run the descriptor carries
// (mvm_fused_batch.h), each run's buffers of that step taken from the descriptor's runs. The descriptor is copied from global memory, as the
// batched bed's kernel copies it; no arena prefetch (the step's tiles use the whole arena). A member count or step the descriptor disagrees
// with fails the run before any barrier (every member reads the same descriptor, so all fail alike); any other failure ends the gang at its
// next crossing. A member takes its place in the gang's crossings at each step's start (mvm_gang::cross_start): its previous step's task
// ended after that step's last crossing, so no member has passed this step's first one (mvm_gang.cuh's contract).
template <bool GRID>   // the grid step's arithmetic: an entry and a call tree of its own (mvm_fused_batch.cuh: run_item)
__device__ __noinline__ void mvm_s_fmlp_batch(const MvmBatchDesc* dp, int member, int members, int step, float* arena, MvmCtx* c) {
  const MvmBatchDesc d = *dp;
  if (members != d.members || member < 0 || member >= members || step < 0 || step >= d.steps || GRID != (d.gHp > 0)) { c->fail(MVM_ERR_BAD_OP, MVM_OP_fmlp_batch); return; }
  unsigned long long at = mvm_gang::cross_start(d.ctl, member, members);
  mvm_fbatch::batch_step<GRID>(member, d, step, arena, at, &c->cnt->err_code, [](int) {});
}

// The ops a fused entry runs besides its fused tasks (mvm_fuse_entry admits nothing else): the steps' data windows and the ordering noops.
// A task of any other op fails the run, as the generic dispatch fails a fused op.
__device__ __noinline__ void mvm_dispatch_fused(int op, const MvmCtx c, const int32_t* a) {
  switch (op) {
    case MVM_OP_noop: break;
#if __has_include("mvm_ops_gpt.def")   // the windows are the GPT pack's op (mvm_ops_list.inc's own test); a build without the pack has no MLP bundle to fuse
    case MVM_OP_select_window: op_select_window(c, a); break;
#endif
    default: c.fail(MVM_ERR_BAD_OP, op);
  }
}

// The kernel entries (mvm_fuse_entry picks one per bundle). `generic` runs every op but the fused ones, which its dispatch refuses, so it
// holds no fused function and a fused task runs only where its own functions are the entry's large callees. `block` runs the one-block
// form (fmlp_slice, fmlp_sgd); `split` (at S_ shares; `split_stream`, the member streaming its weights), `grid` (`grid_virtual`, its members
// carried by fewer blocks) and `batch` their gang op;
// each, through mvm_dispatch_fused, the steps' windows and noops. No fused entry's name contains `mvm_static_kernel`, which the SASS census
// reads the generic kernel by.
enum class MvmSEntry { generic, block, split, grid, batch, batch_grid, split_stream, grid_virtual };

template <MvmSEntry E, int S_ = 0, bool ODX = true>   // ODX: split_stream's member with the owner-side backward compiled in (mvm_fused_split.cuh)
__device__ void mvm_static_worker(const MvmStaticDev& Sref, MvmSShared& sh) {
  const MvmStaticDev S = Sref;   // a register copy of the launch state: fields then read from registers, not through the __grid_constant__ reference (the same copy measured x1.04-1.09 on the dynamic worker)
  const int w = blockIdx.x;
  if (w >= S.nworkers) return;     // (the warm-up launch: nworkers = 0, every block returns at once, the code is loaded)
  const int first = S.offs[w], n = S.offs[w + 1] - first;
  if (n <= 0) {
    if (S.spin_idle && threadIdx.x == 0) { while (ld_acq(S.finished) < S.nactive && !*(volatile int*)&S.cnt->err_code) __nanosleep(1000); }
    return;
  }
  const MvmSRec* prog = S.recs + first;
  if (threadIdx.x == 0) sh.stop = 0;
  mvm_s_fetch(prog, &sh.rec[0]);
  mvm_s_fetch_wait();
  __syncthreads();
  for (int pc = 0; pc < n; ++pc) {
    const MvmSRec* r = &sh.rec[pc & 1];
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * MVM_S_TRACE_WORDS] = mvm_now();
    if (pc + 1 < n) mvm_s_fetch(prog + pc + 1, &sh.rec[(pc + 1) & 1]);   // the next record is copied beside this one's waits and op body
    // a fused op's descriptor: its round trip overlaps this record's waits
    if constexpr (E == MvmSEntry::generic) {}
    else if constexpr (E == MvmSEntry::block) { if (r->op == MVM_OP_fmlp_slice || r->op == MVM_OP_fmlp_sgd) mvm_fmlp::prefetch_desc(S.fused + r->args[0], reinterpret_cast<float*>(mvm_gpt_smem)); }
    else if constexpr (E == MvmSEntry::split || E == MvmSEntry::split_stream) { if (r->op == MVM_OP_fmlp_split) mvm_fsplit::prefetch_desc(S.split + r->args[0], reinterpret_cast<float*>(mvm_gpt_smem)); }
    else if constexpr (E == MvmSEntry::grid || E == MvmSEntry::grid_virtual) { if (r->op == MVM_OP_fmlp_grid) mvm_fgrid::prefetch_desc(S.grid + r->args[0], reinterpret_cast<float*>(mvm_gpt_smem)); }
    const int nw = r->nwait;
    if (nw) {   // only a record with waits needs the hand-over barrier (a plain chain continues directly)
      if (threadIdx.x < nw) {   // one lane of warp 0 per wait: a fan-in of up to MVM_S_WAITS costs one poll, not one per source
        if (!mvm_s_await(S, r->waits[threadIdx.x].worker, r->waits[threadIdx.x].need, r->task)) sh.stop = 1;
      }
      __syncthreads();    // every polling lane's acquire is every thread's (causality through the barrier)
    }
    if (sh.stop) { mvm_s_fetch_wait(); return; }
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * MVM_S_TRACE_WORDS + 1] = mvm_now();
    const int flags = r->flags;
    if (!(flags & MVM_S_WAITONLY)) {
      MvmCtx c{r->bufs, r->nbufs, r->task, S.cnt, S.herr, sh.smem};
      float* const arena = reinterpret_cast<float*>(mvm_gpt_smem);
      if constexpr (E == MvmSEntry::generic) {
        mvm_dispatch<MVM_IN_STATIC>(r->op, c, r->args);   // the static kernel's forms (matmul_tile: no 256x128 fast forms)
      } else if constexpr (E == MvmSEntry::block) {
        if (r->op == MVM_OP_fmlp_slice) mvm_s_fmlp_slice(r->args[1], reinterpret_cast<const int32_t*>(r->bufs[r->args[2]].p), reinterpret_cast<const int32_t*>(r->bufs[r->args[3]].p), arena, S.fuse_phases ? S.fuse_phases + (size_t)w * MVM_FMLP_NPHASES : nullptr, &sh.t_phase, &c);
        else if (r->op == MVM_OP_fmlp_sgd) mvm_s_fmlp_sgd(r->args[1], reinterpret_cast<float*>(r->bufs[r->args[2]].p), arena);
        else mvm_dispatch_fused(r->op, c, r->args);
      } else if constexpr (E == MvmSEntry::split) {
        if (r->op == MVM_OP_fmlp_split) mvm_s_fmlp_split<S_>(r->args[2], r->args[3], reinterpret_cast<const int32_t*>(r->bufs[r->args[4]].p), reinterpret_cast<const int32_t*>(r->bufs[r->args[5]].p), reinterpret_cast<float*>(r->bufs[r->args[6]].p), arena, &c);
        else mvm_dispatch_fused(r->op, c, r->args);
      } else if constexpr (E == MvmSEntry::split_stream) {
        if (r->op == MVM_OP_fmlp_split) mvm_s_fmlp_split_stream<S_, ODX>(r->args[2], r->args[3], reinterpret_cast<const int32_t*>(r->bufs[r->args[4]].p), reinterpret_cast<const int32_t*>(r->bufs[r->args[5]].p), reinterpret_cast<float*>(r->bufs[r->args[6]].p), arena, &c);
        else mvm_dispatch_fused(r->op, c, r->args);
      } else if constexpr (E == MvmSEntry::grid) {
        if (r->op == MVM_OP_fmlp_grid) mvm_s_fmlp_grid(r->args[2], r->args[3], reinterpret_cast<const int32_t*>(r->bufs[r->args[4]].p), reinterpret_cast<const int32_t*>(r->bufs[r->args[5]].p), reinterpret_cast<float*>(r->bufs[r->args[6]].p), arena, &c);
        else mvm_dispatch_fused(r->op, c, r->args);
      } else if constexpr (E == MvmSEntry::grid_virtual) {
        if (r->op == MVM_OP_fmlp_grid) mvm_s_fmlp_grid_virtual(r->args[2], r->args[3], reinterpret_cast<const int32_t*>(r->bufs[r->args[4]].p), reinterpret_cast<const int32_t*>(r->bufs[r->args[5]].p), reinterpret_cast<float*>(r->bufs[r->args[6]].p), arena, &c);
        else mvm_dispatch_fused(r->op, c, r->args);
      } else {
        if (r->op == MVM_OP_fmlp_batch) mvm_s_fmlp_batch<E == MvmSEntry::batch_grid>(S.batch + r->args[0], r->args[2], r->args[3], r->args[4], arena, &c);
        else mvm_dispatch_fused(r->op, c, r->args);
      }
    }
    if (threadIdx.x == 0 && ((pc & 63) == 63) && *(volatile int*)&S.cnt->err_code) sh.stop = 1;   // a failure elsewhere ends the run within 64 records
    if (flags & MVM_S_PUBLISH) __threadfence();    // each warp makes its own op stores visible GPU-wide (mvm_exec.cuh, mvm_run_task)
    mvm_s_fetch_wait();                             // the next record has landed before the barrier that exposes it
    __syncthreads();                                // also exposes sh.stop
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * MVM_S_TRACE_WORDS + 2] = mvm_now();
    if ((flags & MVM_S_PUBLISH) && threadIdx.x == 0) st_rel(S.progress + (size_t)w * MVM_S_LINE_WORDS, r->pos1);   // release: every warp's fence finished before the barrier, so no further fence is needed
    if (S.trace_on && threadIdx.x == 0) S.trace[(size_t)(first + pc) * MVM_S_TRACE_WORDS + 3] = mvm_now();
  }
  if (threadIdx.x == 0) atomicAdd(S.finished, 1);
}

__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_kernel(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::generic>(S, sh);
}
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_block(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::block>(S, sh);
}
template <int S_>
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_split(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::split, S_>(S, sh);
}
template <int S_>
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_split_stream(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::split_stream, S_>(S, sh);
}
// The partials member's streamed entry (a bundle none of whose gangs takes the owner-side dX, mvm_split_odx): split_run without the owner-side
// backward compiled in, so its code is the streamed member's as it was before it: beside that code, a gang of shares of different widths ran 1.6%
// slower on the executor (h440-d2 x8 against the streamed member alone, Nov 15).
template <int S_>
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_split_stream_part(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::split_stream, S_, false>(S, sh);
}
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_grid(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::grid>(S, sh);
}
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_grid_virtual(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::grid_virtual>(S, sh);
}
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_batch(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::batch>(S, sh);
}
__global__ void __launch_bounds__(MVM_THREADS, MVM_MIN_BLOCKS) mvm_static_gang_batch_grid(const __grid_constant__ MvmStaticDev S) {
  __shared__ MvmSShared sh;
  mvm_static_worker<MvmSEntry::batch_grid>(S, sh);
}

// The entry a bundle runs on, as mvm_fuse_entry chose it (nullptr: a share count outside MVM_SPLIT_SHARE_LIST, or outside MVM_SPLIT_STREAM_LIST
// for a streamed member, or a mixed bundle).
using MvmStaticEntryFn = void (*)(const MvmStaticDev);
inline MvmStaticEntryFn mvm_static_entry_fn(const MvmFuseEntry& e) {
  switch (e.kind) {
    case MvmFuseEntry::generic: return mvm_static_kernel;
    case MvmFuseEntry::block: return mvm_static_block;
    case MvmFuseEntry::grid: return mvm_static_gang_grid;
    case MvmFuseEntry::grid_virtual: return mvm_static_gang_grid_virtual;
    case MvmFuseEntry::batch: return mvm_static_gang_batch;
    case MvmFuseEntry::batch_grid: return mvm_static_gang_batch_grid;
    case MvmFuseEntry::split:
      if (e.stream) {   // the streamed member's entries: only the counts it is built for (MVM_SPLIT_STREAM_LIST; the pass streams no other)
        switch (e.S) {
#define X(S_) case S_: return e.odx ? mvm_static_gang_split_stream<S_> : mvm_static_gang_split_stream_part<S_>;
          MVM_SPLIT_STREAM_LIST
#undef X
        }
        return nullptr;
      }
      switch (e.S) {
#define X(S_) case S_: return mvm_static_gang_split<S_>;
        MVM_SPLIT_SHARE_LIST
#undef X
      }
      return nullptr;
    default: return nullptr;
  }
}
#endif  // __CUDACC__
