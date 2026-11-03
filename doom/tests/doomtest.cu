// Standalone GPU Doom check (no scheduler): one 256-thread CTA runs N frames of the REAL engine through the same device functions the
// scheduler ops use (include/kvm_ops_game.cuh), dumping every frame and its state like cpu_run.py so compare.py can diff them.
//   build: nvcc -O2 -arch=sm_80 -rdc=true -I../../include doomtest.cu ../build/doom_dev.o -o doomtest
//   run:   doomtest <doom1.wad> <frames> <outdir> [input.bin]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include "kvm_ops_game.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "[doomtest] CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__device__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

__global__ void run_frames(int n, const float* input, int n_input, float* state, int n_state, float* frames, float* state_hist, unsigned long long* ns) {
  for (int f = 0; f < n; f++) {
    const unsigned long long t0 = gtimer();
    int err = kvm_doom_tick(f, input, n_input, state, n_state);
    __syncthreads();
    if (threadIdx.x == 0) { ns[f] = gtimer() - t0; if (err) printf("[doomtest] tick %d error %d\n", f, err); }
    err = kvm_doom_present(frames + (size_t)f * KVM_DOOM_FB_WORDS, KVM_DOOM_FB_WORDS, 0, DOOM_H, state, n_state, f);
    if (err && threadIdx.x == 0) printf("[doomtest] present %d error %d\n", f, err);
    err = kvm_doom_status(f, state, n_state);
    if (err && threadIdx.x == 0) printf("[doomtest] status %d error %d\n", f, err);
    __syncthreads();
    if (threadIdx.x < 14) state_hist[f * 14 + threadIdx.x] = state[threadIdx.x];
    __syncthreads();
  }
}

__global__ void read_log(char* out, long* n) { for (int i = threadIdx.x; i < (int)sizeof doom_log; i += blockDim.x) out[i] = doom_log[i]; if (!threadIdx.x) *n = doom_log_n; }

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: doomtest <doom1.wad> <frames> <outdir> [input.bin]\n"); return 2; }
  const int n = atoi(argv[2]);
  const std::string out = argv[3];
  mkdir(out.c_str(), 0755);
  if (!kvm_doom_host_init(argv[1])) return 1;
  std::vector<uint32_t> in_words;
  if (argc > 4) { FILE* f = fopen(argv[4], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[4]); return 1; } uint32_t w; while (fread(&w, 4, 1, f) == 1) in_words.push_back(w); fclose(f); in_words.insert(in_words.begin(), KVM_DOOM_INPUT_MAGIC); }
  else in_words.assign(1, 0);                                                          // no magic: the scripted player plays
  const int n_state = KVM_DOOM_ST_HASH + 2 * n;
  float *d_in, *d_state, *d_frames, *d_hist; unsigned long long* d_ns;
  CK(cudaMalloc(&d_in, in_words.size() * 4)); CK(cudaMemcpy(d_in, in_words.data(), in_words.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_state, n_state * 4)); CK(cudaMemset(d_state, 0, n_state * 4));
  CK(cudaMalloc(&d_frames, (size_t)n * KVM_DOOM_FB_BYTES)); CK(cudaMalloc(&d_hist, n * 14 * 4)); CK(cudaMalloc(&d_ns, n * 8));
  fprintf(stderr, "[doomtest] %d frames, stack %d B/thread, %zu-word input\n", n, KVM_DOOM_STACK_BYTES, in_words.size());
  const auto t0 = std::chrono::steady_clock::now();
  run_frames<<<1, 256>>>(n, d_in, (int)in_words.size(), d_state, n_state, d_frames, d_hist, d_ns);
  const cudaError_t e = cudaDeviceSynchronize();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  static char logbuf[1 << 16]; long logn = 0;
  { char* d_log; long* d_n; CK(cudaMalloc(&d_log, sizeof logbuf)); CK(cudaMalloc(&d_n, 8)); read_log<<<1, 256>>>(d_log, d_n);
    CK(cudaMemcpy(logbuf, d_log, sizeof logbuf, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&logn, d_n, 8, cudaMemcpyDeviceToHost)); }
  fprintf(stderr, "[doomtest] kernel %s in %.3f s (%.2f ms/frame incl. boot)\n--- engine log ---\n%s\n------------------\n", cudaGetErrorString(e), secs, secs * 1e3 / n, logbuf);
  if (e != cudaSuccess) return 1;
  std::vector<float> state(n_state), hist(n * 14); std::vector<unsigned long long> ns(n);
  CK(cudaMemcpy(state.data(), d_state, n_state * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hist.data(), d_hist, hist.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ns.data(), d_ns, n * 8, cudaMemcpyDeviceToHost));
  std::vector<unsigned char> frames((size_t)n * KVM_DOOM_FB_BYTES);
  CK(cudaMemcpy(frames.data(), d_frames, frames.size(), cudaMemcpyDeviceToHost));
  FILE* fb = fopen((out + "/frames.bin").c_str(), "wb"); fwrite(frames.data(), 1, frames.size(), fb); fclose(fb);
  static const char* names[14] = {"booted", "tics", "gamestate", "health", "armor", "kills", "monsters", "items", "secrets", "deaths", "level_finished_tic", "x", "y", "waypoint"};
  FILE* js = fopen((out + "/summary.json").c_str(), "w");
  fprintf(js, "{\"frames\": %d, \"hashes\": [", n);
  for (int f = 0; f < n; f++) { uint32_t a, b; memcpy(&a, &state[KVM_DOOM_ST_HASH + 2 * f], 4); memcpy(&b, &state[KVM_DOOM_ST_HASH + 2 * f + 1], 4); fprintf(js, "%s[%u, %u]", f ? ", " : "", a, b); }
  fprintf(js, "], \"states\": [");
  for (int f = 0; f < n; f++) { fprintf(js, "%s[", f ? ", " : ""); for (int i = 0; i < 14; i++) { int v; memcpy(&v, &hist[f * 14 + i], 4); fprintf(js, "%s%d", i ? ", " : "", v); } fprintf(js, "]"); }
  fprintf(js, "], \"state_names\": [");
  for (int i = 0; i < 14; i++) fprintf(js, "%s\"%s\"", i ? ", " : "", names[i]);
  fprintf(js, "], \"tick_ns\": [");
  for (int f = 0; f < n; f++) fprintf(js, "%s%llu", f ? ", " : "", ns[f]);
  fprintf(js, "]}\n"); fclose(js);
  std::vector<unsigned long long> s = ns; std::sort(s.begin() + 1, s.end());
  fprintf(stderr, "[doomtest] tick time ns: boot(frame 0) %llu, median %llu, p99 %llu, max %llu\n", ns[0], s[s.size() / 2], s[s.size() * 99 / 100], s.back());
  fprintf(stderr, "[doomtest] wrote %s/{frames.bin,summary.json}\n", out.c_str());
  return 0;
}
