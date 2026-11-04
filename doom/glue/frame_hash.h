// Frame fingerprint shared by the CPU oracle (C) and the GPU ops (CUDA): 64 bits over the 64000 palette indices and the 256 palette
// words. A sum and an xor of per-element mixes, so the GPU can reduce it in any order and still equal the CPU's.
#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define DH_FN __host__ __device__ static inline
#else
#define DH_FN static inline
#endif
#define DOOM_W 320
#define DOOM_H 200
#define DOOM_PIXELS (DOOM_W * DOOM_H)
DH_FN uint32_t dh_mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
// element i (0..63999 pixels, 64000..64255 palette words) with value v
DH_FN uint32_t dh_term(uint32_t i, uint32_t v) { return dh_mix(i * 0x9E3779B1U ^ dh_mix(v + 0x632BE5ABU)); }
