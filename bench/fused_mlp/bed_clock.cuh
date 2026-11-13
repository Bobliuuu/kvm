#pragma once
// The fused MLP beds' clocks (split_bed.cu, grid_bed.cu): block 0's phase stamps, its %globaltimer around the counted steps, each block's SM.
#include <cstdint>

struct Timing { long long* phase; unsigned long long* gtime; unsigned* smid; };   // [NPH] cycles summed over steps (block 0); [2] its %globaltimer; [blocks] each block's SM
__device__ __forceinline__ unsigned sm_id() { unsigned r; asm volatile("mov.u32 %0, %%smid;" : "=r"(r)); return r; }
__device__ __forceinline__ unsigned long long global_ns() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }

// Block 0's phase clock: every warp stamps its own end of a phase (a shared atomic max) and thread 0 turns the stamps into durations one step later,
// so a phase ends at its last warp's end (thread 0's clock read after a barrier runs when warp 0 arrives: sm_80's barrier holds a warp only at its
// next memory access). The stamps alternate between two rows by step parity; no barrier is added.
struct Stamps {
  unsigned long long* row;
  __device__ __forceinline__ void operator()(int i) const { if (blockIdx.x == 0 && (threadIdx.x & 31) == 0) atomicMax(row + i, (unsigned long long)clock64()); }
};
