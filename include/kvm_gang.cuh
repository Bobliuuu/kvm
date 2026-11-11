#pragma once
// Gang barrier: the blocks of one gang (a fused op's members, placed by the static planner on distinct workers at one point of every
// program: kvm_static_plan; or a fused step's gang) meet here, and every global store a member made before it is visible to every member
// after it. Two exact forms keep one contract, and `cross` (below) picks one by the gang's size:
//  * the flag lines (`barrier_lines`): each member publishes its count of crossings in its own word, and its warps poll every member's word;
//  * the counted barrier (`barrier_counted`): one 64-bit count of arrivals, which a member's thread 0 adds itself to and polls.
// The contract:
//  * no reset between uses or between runs of a bundle: the control words are zeroed once, and a member takes its place in the gang's
//    crossings from them at the start of each run (`cross_start`);
//  * one gang's control serves every crossing of that gang; two gangs that may run at the same time need two;
//  * `abort` (optional) is the run's error word: a member waiting here reads it every 4,096 polls and gives up once it is set, so a member
//    that failed and never arrives ends the gang's run instead of hanging it. The barrier returns false then (every thread of the block
//    gets the same answer), and the caller must return without another crossing: the words are left mid-crossing, and the failed run's
//    buffers are not reused;
//  * the answer is the block's OR at the exit barrier (__syncthreads_or), so neither form holds static shared memory: every kernel entry
//    that calls one keeps the static shared memory its worker declares, which the fused steps' budget assumes.
// Ordering (PTX memory model, gpu scope): __syncthreads, then thread 0's __threadfence, orders the block's prior stores before its arrival
// (a release); a waiter's acquire load that sees every arrival, then __syncthreads, makes every member's stores visible to its block.
#include <cstdint>

namespace kvm_gang {

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_release(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void red_release_add(unsigned long long* p, unsigned long long v) {
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

// The flag-line form. Member m's thread 0 publishes its epoch, the count of its crossings, in word m of `lines` (line_words(members)
// words, 128-byte aligned, zeroed once), and warp w polls line w, lane l acquiring member 32 w + l's word, until every member's epoch has
// reached its own; the block barrier then hands that to every thread. One round trip to L2 a crossing and no atomic, but every member's
// warps poll every line, 4 sectors a line a poll, so its cost grows with members x lines. The words need no reset: every member of a
// completed run crosses equally often, so they stay equal, and a member starts each run from its own word (`lines_epoch`). Every thread
// of the block calls it, and the block has a warp for each line: members <= 32 * (blockDim.x / 32).
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

// The counted form. The gang's control is one 64-bit count of arrivals in its first two words (8-byte aligned), zeroed once and never
// reset:
//  * a member adds itself with a release reduction (`red.release.gpu.add`: no value comes back, so the arrival costs no round trip);
//  * its thread 0 then polls the count with acquire loads until it reaches this crossing's target, `members` past the last one.
// The reductions on the count form one release sequence, so the acquire load that reads a value at or past the target synchronizes with
// every member's release. A member's first target in a run comes from the count it reads before its first arrival (`counted_base`),
// rounded down to a multiple of `members`. That read sees c0 + k: c0 a multiple of `members` (every completed crossing added `members`)
// and k arrivals at the crossing this member has not reached yet (k < members: no member passes that crossing before this one arrives),
// so rounding down gives c0 whoever else has started. A run that ends cleanly leaves a multiple of `members`, so consecutive steps and
// runs need no reset; 2^64 arrivals never wrap.
__device__ __forceinline__ unsigned long long counted_base(const unsigned* ctl, int members) {
    const unsigned long long v = *reinterpret_cast<const volatile unsigned long long*>(ctl);
    return v - v % (unsigned long long)members;
}
__device__ __forceinline__ bool barrier_counted(unsigned* ctl, int members, unsigned long long& target, const int* abort = nullptr) {
    unsigned long long* const count = reinterpret_cast<unsigned long long*>(ctl);
    int gave = 0;
    __syncthreads();
    target += (unsigned long long)members;
    if (threadIdx.x == 0) {
        __threadfence();
        red_release_add(count, 1ull);
        for (unsigned spin = 1; (long long)(ld_acquire(count) - target) < 0; ++spin)
            if (abort && (spin & 4095u) == 0 && *(const volatile int*)abort) { gave = 1; break; }
    }
    return !__syncthreads_or(gave);
}

// The gang's crossing: the flag lines up to LinesMax members (by default one line, 32), the counted barrier past them. A flag-line
// crossing is one round trip, but past one line every member's warps poll members / 32 lines, and the waiters' polls slow the members
// still working. On the A100 (sm_80, 108 SMs) a flag-line crossing alone took 1,350 ns at 32 members and 3,465 at 108
// (results/2026-11-11-gang-cross/); inside the fused steps the flag lines won at 32 members and the counted barrier from 60 up
// (results/2026-11-11-gang-counted/). One allocation serves both forms: line_words(members) words, zeroed once and 8-byte aligned (the
// counted form uses the first two). A member's place in the crossings (`at`) comes from cross_start: its flag-line epoch, or its count's
// base. A caller's block needs a warp for each line of LinesMax members: LinesMax <= 32 * (blockDim.x / 32).
#ifndef KVM_GANG_LINES_MAX
#define KVM_GANG_LINES_MAX 32
#endif
// The form `cross` takes for a gang of `members`: the counted barrier past LinesMax members, else the flag lines. A kernel that picks its
// form once at launch (an instance per form, chosen on the host by this) calls that form directly at every crossing, where `cross` would
// carry both forms' code behind a run-time branch at each one.
template <int LinesMax = KVM_GANG_LINES_MAX>
__host__ __device__ constexpr bool counted_form(int members) { return members > LinesMax; }
template <int LinesMax = KVM_GANG_LINES_MAX>
__device__ __forceinline__ unsigned long long cross_start(const unsigned* ctl, int member, int members) {
    return members <= LinesMax ? (unsigned long long)lines_epoch(ctl, member) : counted_base(ctl, members);
}
template <int LinesMax = KVM_GANG_LINES_MAX>
__device__ __forceinline__ bool cross(unsigned* ctl, int member, int members, unsigned long long& at, const int* abort = nullptr) {
    if (members <= LinesMax) {
        unsigned epoch = (unsigned)at;
        const bool ok = barrier_lines(ctl, member, members, epoch, abort);
        at = epoch;
        return ok;
    }
    return barrier_counted(ctl, members, at, abort);
}

}  // namespace kvm_gang
