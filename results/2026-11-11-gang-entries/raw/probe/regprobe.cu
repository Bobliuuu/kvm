// Register probe: the static kernel's fused part, without the executor. The four fused wrappers exactly as kvm_static.cuh has them (a
// stand-in context in place of KvmCtx: a noinline fail call, as kvm_fail is), called from a worker loop that holds its state across the calls.
// PROBE_SLICE, PROBE_SPLIT and PROBE_GRID pick which wrappers the loop calls; probe_grid_inline is the bed's form (grid_run in the kernel).
#include <cstdio>
#include <cstdint>
#include "kvm_fused_grid.cuh"

#ifndef PROBE_SLICE
#define PROBE_SLICE 1
#endif
#ifndef PROBE_SPLIT
#define PROBE_SPLIT 1
#endif
#ifndef PROBE_GRID
#define PROBE_GRID 1
#endif

extern __shared__ __align__(128) unsigned char kvm_gpt_smem[];
struct ProbeCnt { int err_code; int slot, a, b; };
__device__ __noinline__ void probe_fail(ProbeCnt* c, int code, int v) {
  if (atomicCAS(&c->err_code, 0, code) == 0) { c->slot = blockIdx.x; c->a = v; }
}
struct ProbeCtx { ProbeCnt* cnt; __device__ void fail(int code, int v) const { probe_fail(cnt, code, v); } };

#if PROBE_SLICE
__device__ __noinline__ void kvm_s_fmlp_slice(int sc, const int32_t* ids, const int32_t* tgt, float* arena, long long* ph, long long* ph_last, ProbeCtx* c) {
  kvm_fmlp::slice_run(sc, ids, tgt, nullptr, arena, kvm_fmlp::PhaseTimer{ph, ph_last}, [c](int code, int v) { c->fail(code, v); });
}
__device__ __noinline__ void kvm_s_fmlp_sgd(int owner, float* loss, float* arena) { kvm_fmlp::sgd_run(owner, loss, arena); }
#endif
#if PROBE_SPLIT
__device__ __noinline__ void kvm_s_fmlp_split(int member, int members, const int32_t* ids, const int32_t* tgt, float* loss, float* arena, ProbeCtx* c) {
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  const KvmSplitDesc& d = *reinterpret_cast<const KvmSplitDesc*>(arena);
  if (d.Q != KVM_SPLIT_Q || members != d.Q * d.S || member < 0 || member >= members) { c->fail(21, 192); return; }
#ifdef PROBE_SPLIT_S
  if (d.S == PROBE_SPLIT_S) { kvm_fsplit::split_run<PROBE_SPLIT_S, KVM_SPLIT_Q>(member, ids, tgt, loss, arena, &c->cnt->err_code, kvm_fsplit::NoPhases{}); return; }
  c->fail(21, 192);
#else
  switch (d.S) {
#define X(S_) case S_: kvm_fsplit::split_run<S_, KVM_SPLIT_Q>(member, ids, tgt, loss, arena, &c->cnt->err_code, kvm_fsplit::NoPhases{}); return;
    KVM_SPLIT_SHARE_LIST
#undef X
    default: c->fail(21, 192);
  }
#endif
}
#endif
#if PROBE_GRID
__device__ __noinline__ void kvm_s_fmlp_grid(int member, int members, const int32_t* ids, const int32_t* tgt, float* loss, float* arena, ProbeCtx* c) {
  kvm_fmlp::cp_commit_wait();
  __syncthreads();
  const KvmGridDesc& d = *reinterpret_cast<const KvmGridDesc*>(arena);
  if (members != d.P || member < 0 || member >= members) { c->fail(21, 194); return; }
  kvm_fgrid::grid_run(member, ids, tgt, loss, arena, &c->cnt->err_code, kvm_fgrid::NoPhases{});
}
#endif

struct PRec { int op, a0, a1, pad; const int32_t* ids; const int32_t* tgt; float* loss; };
// The worker loop's shape: its program, its watermark and the context stay live across every call, as kvm_static_worker's do.
__global__ void __launch_bounds__(256, 1) probe_static(const PRec* recs, const int* offs, int* progress, ProbeCnt* cnt, long long* ph) {
  const int w = blockIdx.x, first = offs[w], n = offs[w + 1] - first;
  float* const arena = reinterpret_cast<float*>(kvm_gpt_smem);
  ProbeCtx c{cnt};
  for (int pc = 0; pc < n; ++pc) {
    const PRec r = recs[first + pc];
#if PROBE_SLICE
    if (r.op == 1) kvm_s_fmlp_slice(r.a0, r.ids, r.tgt, arena, ph + 2 * w, ph + 2 * w + 1, &c);
    else if (r.op == 2) kvm_s_fmlp_sgd(r.a0, r.loss, arena);
#endif
#if PROBE_SPLIT
    if (r.op == 3) kvm_s_fmlp_split(r.a0, r.a1, r.ids, r.tgt, r.loss, arena, &c);
#endif
#if PROBE_GRID
    if (r.op == 4) kvm_s_fmlp_grid(r.a0, r.a1, r.ids, r.tgt, r.loss, arena, &c);
#endif
    __syncthreads();
    if (threadIdx.x == 0) progress[w * 32] = pc + 1;
  }
}

// The bed's form: grid_run inlined into a kernel of its own.
__global__ void __launch_bounds__(256, 1) probe_grid_inline(const int32_t* ids, const int32_t* tgt, float* loss, int* abort) {
  float* const arena = reinterpret_cast<float*>(kvm_gpt_smem);
  kvm_fgrid::grid_run(blockIdx.x, ids, tgt, loss, arena, abort, kvm_fgrid::NoPhases{});
}

int main() { printf("regprobe: SASS only\n"); return 0; }
