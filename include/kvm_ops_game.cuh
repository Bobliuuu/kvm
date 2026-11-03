#pragma once
// Device side of the game operators in kvm_ops_game.def: the REAL Doom engine (doomgeneric, built by doom/build.sh into
// doom/build/doom_dev.o) running as ordinary scheduler tasks.
//
// ===================== EXECUTOR REQUIREMENTS (read by agent B / whoever builds the dispatcher) =====================
//  1. blockDim.x >= 256 (the engine uses exactly threads 0..255 of the CTA: thread 0 is the game thread, 1..255 draw columns/spans
//     behind named barrier 2; barrier 0 stays free for __syncthreads). Every thread of the CTA must call the op (block-cooperative).
//  2. Link: nvcc -rdc=true <your.cu> doom/build/doom_dev.o (the engine is a relocatable device object; its globals are device-global,
//     so there is exactly ONE Doom instance per process). Build it with doom/build.sh (ARCH=sm_80 default).
//  3. Stack: the engine recurses (BSP walk); cudaDeviceSetLimit(cudaLimitStackSize, KVM_DOOM_STACK_BYTES) before the launch.
//     kvm_doom_host_init() does this and uploads the WAD. The limit is per resident thread, so it costs
//     SMs * threads/SM * KVM_DOOM_STACK_BYTES of device memory.
//  4. ONE doom_tick in flight at a time, in order: tick f+1 depends on tick f (chain with DAG edges). doom_present(f)/doom_status(f)
//     read the screen of the LAST completed tick, so they must be ordered after tick f and before tick f+1 (edges both ways, or
//     present f -> tick f+1). The ops check state tic == frame+1 and fail loudly (KVM_ERR_USER) when ordering is violated.
//  5. Cross-CTA visibility is the executor's contract (release at publish / acquire at pop, fence after the op): this file only uses
//     __ldcg for data other CTAs wrote, and plain stores for what it produces.
//  6. Needs no more than the generic arguments: buffers are f32 table entries; ops reinterpret words. Scheduling: the 35 Hz pacing is
//     deadline metadata only (deadline_us on the task, doom/make_workload.py); the scheduler needs no knowledge of Doom.
//  7. A tick takes milliseconds on one CTA (serial game logic), far longer than a GEMM tile: the CTA is occupied for that time.
//
// ===================== buffer layouts (32-bit words) =====================
//  input : word 0 = KVM_DOOM_INPUT_MAGIC, word 1+f = input for tic f: bits 0..7 forwardmove (int8), 8..15 sidemove (int8),
//          16..23 turn (int8, x32 = angleturn), 24..31 buttons (BT_ATTACK=1, BT_USE=2). 0xFFFFFFFF = scripted player for that tic.
//          Without the magic (a zero-filled buffer, or f beyond its size) the scripted player plays: no cheats, the demo plays itself.
//  state : [0] booted [1] tics run [2] gamestate (0 level, 1 intermission) [3] health [4] armor [5] kills [6] monsters [7] items
//          [8] secrets [9] deaths [10] tic the level finished (-1) [11] x [12] y [13] route waypoint, [14..15] reserved,
//          then 2 words per frame from KVM_DOOM_ST_HASH: sum, xor of the frame fingerprint (doom/glue/frame_hash.h).
//  fb    : >= KVM_DOOM_FB_WORDS words = 64000 bytes of palette indices (row-major 320x200) + 256 RGBA/XRGB palette words at word 16000.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "../doom/glue/frame_hash.h"

#define KVM_DOOM_INPUT_MAGIC 0x4B444D49u
#define KVM_DOOM_INPUT_SCRIPTED 0xFFFFFFFFu
#define KVM_DOOM_FB_BYTES (DOOM_PIXELS + 1024)
#define KVM_DOOM_FB_WORDS (KVM_DOOM_FB_BYTES / 4)
#define KVM_DOOM_PAL_WORD (DOOM_PIXELS / 4)
#define KVM_DOOM_ST_HEADER 16
#define KVM_DOOM_ST_BOOTED 0
#define KVM_DOOM_ST_TICS 1
#define KVM_DOOM_ST_HASH KVM_DOOM_ST_HEADER
#define KVM_DOOM_THREADS 256
#define KVM_DOOM_STACK_BYTES (16 * 1024)

extern "C" {
__device__ void gpu_doom_boot(const unsigned char* wad, long wad_len);
__device__ void gpu_doom_tic(uint32_t input);
__device__ void gpu_doom_worker(int tid);
__device__ void gpu_doom_abort_tic();
__device__ void doom_state(int32_t* out);
extern __device__ unsigned char* I_VideoBuffer;
extern __device__ uint32_t colors[256];
extern __device__ char doom_log[1 << 16];
extern __device__ volatile long doom_log_n;
}

__device__ const unsigned char* g_kvm_doom_wad;   // uploaded by kvm_doom_host_init()
__device__ long g_kvm_doom_wad_len;

enum KvmDoomStatus { KVM_DOOM_OK = 0, KVM_DOOM_BAD_ORDER = 1, KVM_DOOM_NO_WAD = 2, KVM_DOOM_SMALL_BUF = 3 };

// All threads of the CTA call; the return value is meaningful on thread 0 only (workers return KVM_DOOM_OK).
__device__ inline int kvm_doom_tick(int frame, const float* input, int n_input, float* state, int n_state) {
  const int tid = threadIdx.x;
  if (tid >= KVM_DOOM_THREADS) return KVM_DOOM_OK;
  if (tid != 0) { gpu_doom_worker(tid); return KVM_DOOM_OK; }
  int err = KVM_DOOM_OK;
  if (n_state < KVM_DOOM_ST_HEADER) err = KVM_DOOM_SMALL_BUF;
  else if (frame == 0 && g_kvm_doom_wad == nullptr) err = KVM_DOOM_NO_WAD;
  else if (frame != 0 && __float_as_int(__ldcg(state + KVM_DOOM_ST_TICS)) != frame) err = KVM_DOOM_BAD_ORDER;
  if (err != KVM_DOOM_OK) { gpu_doom_abort_tic(); return err; }
  if (frame == 0) gpu_doom_boot(g_kvm_doom_wad, g_kvm_doom_wad_len);
  uint32_t in = KVM_DOOM_INPUT_SCRIPTED;
  if (input && n_input > frame + 1 && __float_as_uint(__ldcg(input)) == KVM_DOOM_INPUT_MAGIC) in = __float_as_uint(__ldcg(input + frame + 1));
  gpu_doom_tic(in);
  int32_t st[16] = {};
  doom_state(st);
  for (int i = 0; i < 14; i++) state[i] = __int_as_float(st[i]);
  return KVM_DOOM_OK;
}

// All threads of the CTA call. Copies screen rows [row0, row0+rows) and, with row0 == 0, the palette.
// With state != nullptr it also checks that tick `frame` is the last one completed.
__device__ inline int kvm_doom_present(float* fb, int n_fb, int row0, int rows, const float* state, int n_state, int frame) {
  if (n_fb < KVM_DOOM_FB_WORDS || row0 < 0 || rows < 0 || row0 + rows > DOOM_H) return KVM_DOOM_SMALL_BUF;
  if (state && frame >= 0 && __float_as_int(__ldcg(state + KVM_DOOM_ST_TICS)) != frame + 1) return KVM_DOOM_BAD_ORDER;
  const unsigned char* vb = I_VideoBuffer;
  unsigned char* out = reinterpret_cast<unsigned char*>(fb);
  for (int i = row0 * DOOM_W + threadIdx.x; i < (row0 + rows) * DOOM_W; i += blockDim.x) out[i] = __ldcg(vb + i);
  if (row0 == 0 && threadIdx.x < 256) fb[KVM_DOOM_PAL_WORD + threadIdx.x] = __uint_as_float(__ldcg(colors + threadIdx.x));
  return KVM_DOOM_OK;
}

// All threads of the CTA call (any blockDim <= 1024). Records the frame fingerprint of the current screen in state history slot `frame`.
__device__ inline int kvm_doom_status(int frame, float* state, int n_state) {
  __shared__ uint32_t s_sum[32], s_xor[32];
  if (frame < 0 || n_state < KVM_DOOM_ST_HASH + 2 * (frame + 1)) return KVM_DOOM_SMALL_BUF;
  if (__float_as_int(__ldcg(state + KVM_DOOM_ST_TICS)) != frame + 1) return KVM_DOOM_BAD_ORDER;
  const unsigned char* vb = I_VideoBuffer;
  uint32_t s = 0, x = 0;
  for (int i = threadIdx.x; i < DOOM_PIXELS + 256; i += blockDim.x) {
    const uint32_t v = i < DOOM_PIXELS ? __ldcg(vb + i) : __ldcg(colors + (i - DOOM_PIXELS));
    const uint32_t t = dh_term(i, v);
    s += t; x ^= t;
  }
  for (int o = 16; o; o >>= 1) { s += __shfl_down_sync(0xffffffffu, s, o); x ^= __shfl_down_sync(0xffffffffu, x, o); }
  if ((threadIdx.x & 31) == 0) { s_sum[threadIdx.x >> 5] = s; s_xor[threadIdx.x >> 5] = x; }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t ts = 0, tx = 0;
    for (int w = 0; w < (int)((blockDim.x + 31) >> 5); w++) { ts += s_sum[w]; tx ^= s_xor[w]; }
    state[KVM_DOOM_ST_HASH + 2 * frame] = __uint_as_float(ts);
    state[KVM_DOOM_ST_HASH + 2 * frame + 1] = __uint_as_float(tx);
  }
  __syncthreads();
  return KVM_DOOM_OK;
}

// ---------------------------------------------------------------- host helper
// Uploads the WAD and sets the stack limit. Call once before launching any kernel that runs doom_tick. Returns false (logged) on failure.
inline bool kvm_doom_host_init(const char* wad_path, int stack_bytes = KVM_DOOM_STACK_BYTES) {
  FILE* f = fopen(wad_path, "rb");
  if (!f) { fprintf(stderr, "[doom] cannot open WAD %s\n", wad_path); return false; }
  fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<unsigned char> wad(n);
  const bool ok = fread(wad.data(), 1, n, f) == (size_t)n;
  fclose(f);
  if (!ok) { fprintf(stderr, "[doom] short read of %s\n", wad_path); return false; }
  unsigned char* d = nullptr;
  cudaError_t e = cudaMalloc(&d, n);
  if (e == cudaSuccess) e = cudaMemcpy(d, wad.data(), n, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_kvm_doom_wad, &d, sizeof d);
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_kvm_doom_wad_len, &n, sizeof n);
  if (e == cudaSuccess) e = cudaDeviceSetLimit(cudaLimitStackSize, stack_bytes);
  if (e != cudaSuccess) { fprintf(stderr, "[doom] host init failed: %s\n", cudaGetErrorString(e)); return false; }
  fprintf(stderr, "[doom] WAD %s (%ld bytes) uploaded, stack limit %d bytes/thread\n", wad_path, n, stack_bytes);
  return true;
}

// ---------------------------------------------------------------- executor wrappers (active only inside kvm_exec.cuh)
#ifdef KVM_OP
#ifndef KVM_ERR_USER
#define KVM_ERR_USER KVM_ERR_BOUNDS
#endif
KVM_OP(doom_tick) {
  KvmView in = ARG_BUF(1), st = ARG_BUF(2);
  const int err = kvm_doom_tick(ARG_I32(0), in.p, in.rows * in.cols, st.p, st.rows * st.cols);
  if (err) c.fail(KVM_ERR_USER, 128, err);
}
KVM_OP(doom_present) {
  KvmView fb = ARG_BUF(0);
  const int err = kvm_doom_present(fb.p, fb.rows * fb.cols, ARG_I32(1), ARG_I32(2), nullptr, 0, -1);
  if (err) c.fail(KVM_ERR_USER, 129, err);
}
KVM_OP(doom_status) {
  KvmView st = ARG_BUF(1);
  const int err = kvm_doom_status(ARG_I32(0), st.p, st.rows * st.cols);
  if (err) c.fail(KVM_ERR_USER, 130, err);
}
#endif
