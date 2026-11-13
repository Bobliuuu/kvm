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
// The answer is the block's OR at the exit barrier (__syncthreads_or), so neither barrier holds static shared memory: every kernel entry
// that calls one keeps the static shared memory its worker declares, which the fused steps' budget assumes.
__device__ __forceinline__ bool barrier(unsigned* ctl, int members, const int* abort = nullptr) {
    int gave = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned* count = ctl;
        unsigned* gen = ctl + 1;
        const unsigned g0 = ld_acquire(gen);
        if (atom_add_acq_rel(count, 1u) + 1u == (unsigned)members) {
            st_relaxed(count, 0u);          // ordered before the publish by the release below
            st_release(gen, g0 + 1u);
        } else {
            for (unsigned spin = 1; ld_acquire(gen) == g0; ++spin)
                if (abort && (spin & 4095u) == 0 && *(const volatile int*)abort) { gave = 1; break; }
        }
    }
    return !__syncthreads_or(gave);
}

// The flag-line form, for gangs whose barrier is on their critical path. Member m's thread 0 publishes its epoch, the count of its
// crossings, in word m of `lines` (line_words(members) words, 128-byte aligned, zeroed once), and warp w polls line w, lane l
// acquiring member 32 w + l's word, until every member's epoch has reached its own; the block barrier then hands that to every thread. One
// round trip to L2 a crossing and no atomic, where `barrier` waits for an atomic's round trip, a release and a poll: on an RTX 5070 a
// crossing took 710 ns at 32 members against `barrier`'s 1,194. The words need no reset: every member of a completed run crosses equally
// often, so they stay equal, and a member starts each run from its own word (`lines_epoch`). `abort` is read every 4,096 polls, as
// `barrier` reads it; false means the run failed (every thread of the block gets the same answer), and the caller returns without
// another crossing. Every thread of the block calls it, and the block has a warp for each line: members <= 32 * (blockDim.x / 32).
__host__ __device__ constexpr int line_words(int members) { return (members + 31) / 32 * 32; }
__device__ __forceinline__ unsigned lines_epoch(const unsigned* lines, int member) { return *(const volatile unsigned*)(lines + member); }
__device__ __forceinline__ bool barrier_lines(unsigned* lines, int member, int members, unsigned& epoch, const int* abort = nullptr) {
    __syncthreads();
    ++epoch;
    if (threadIdx.x == 0) {
        __threadfence();
        st_release(lines + member, epoch);
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int gave = 0;
    if (warp < (members + 31) / 32) {
        const int m = warp * 32 + lane;
        for (unsigned spin = 1;; ++spin) {
            const unsigned v = m < members ? ld_acquire(lines + m) : epoch;
            if (__all_sync(0xffffffffu, (int)(v - epoch) >= 0)) break;
            if ((spin & 4095u) == 0 && __any_sync(0xffffffffu, abort != nullptr && *(const volatile int*)abort != 0)) {
                gave = 1;
                break;
            }
        }
    }
    return !__syncthreads_or(gave);
}

}  // namespace kvm_gang
