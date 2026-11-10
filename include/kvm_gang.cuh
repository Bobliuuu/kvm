#pragma once
// Gang barrier: the blocks of one gang (a fused op's members, placed by the static planner on distinct workers at one point of every
// program: kvm_static_plan) meet here, and every global store a member made before it is visible to every member after it.
//
// The barrier is sense-reversing, so it needs no reset between uses or between runs of a bundle:
//  * `ctl` holds two words: the count of members arrived at the current crossing, and the crossing's generation;
//  * a member reads the generation, then adds itself to the count;
//  * the last to arrive zeroes the count and then publishes the next generation (a release store);
//  * the others spin until the generation moves (acquire loads).
// The count is zero between crossings, so a run that ends cleanly leaves it ready for the next one. One `ctl` serves every crossing of
// one gang. Two gangs that may run at the same time need two `ctl`s.
//
// Why the read comes before the add: the generation can move only after this member's own add (the last arrival needs it), so the value
// read before the add is the crossing's own, and a member never waits for a generation that has already gone by.
// Ordering (PTX memory model, gpu scope):
//  * __syncthreads, then thread 0's __threadfence, orders the block's prior stores before its arrival;
//  * the arrival is an acq_rel atomic, and the publish a release store after the count's reset;
//  * a waiter's acquire load of the new generation, then __syncthreads, makes every member's stores visible to its block.
// Same form as the grid barrier measured at 1.35-1.49 us across 2-108 blocks on the A100.
#include <cstdint>

namespace kvm_gang {

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_relaxed(unsigned* p, unsigned v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) { unsigned o; asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }

// Every thread of the block calls it; `members` is the gang's size (>= 1); `ctl` points at the gang's two control words in global memory.
// `abort` (optional) is the run's error word: a member waiting at the barrier reads it every 4,096 polls and gives up once it is set, so a member
// that failed and never arrives ends the gang's run instead of hanging it. Returns false then (every thread of the block gets the same answer),
// and the caller must return without another crossing: the words are left mid-crossing, and the failed run's buffers are not reused.
__device__ __forceinline__ bool barrier(unsigned* ctl, int members, const int* abort = nullptr) {
    __shared__ int gave_up;
    __syncthreads();
    if (threadIdx.x == 0) {
        gave_up = 0;
        __threadfence();
        unsigned* count = ctl;
        unsigned* gen = ctl + 1;
        const unsigned g0 = ld_acquire(gen);
        if (atom_add_acq_rel(count, 1u) + 1u == (unsigned)members) {
            st_relaxed(count, 0u);          // ordered before the publish by the release below
            st_release(gen, g0 + 1u);
        } else {
            for (unsigned spin = 1; ld_acquire(gen) == g0; ++spin)
                if (abort && (spin & 4095u) == 0 && *(const volatile int*)abort) { gave_up = 1; break; }
        }
    }
    __syncthreads();
    return !gave_up;
}

}  // namespace kvm_gang
