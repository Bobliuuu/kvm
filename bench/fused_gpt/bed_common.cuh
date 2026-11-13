#pragma once
// What the fused GPT beds share (fgpt_bed.cu runs the fused step, f2d_bed.cu runs F): a workload that fgpt_bed.py wrote, loaded into device
// memory as a bundle's buffers would be (KvmFgptDesc's shape, hyperparameters, parameter slots with their moments and gradients, streams,
// step counter and loss buffers), the warm-up, the phase clock and the outputs fgpt_bed.py judges. Each bed adds its own form's buffers and
// launch. A bed defines BED_NAME, the prefix of its messages, before including this.
//
// The inputs (DIR/meta.txt: `key value` lines and `slot i rows cols wd` a parameter; DIR/{p,m,v}.bin in slot order; DIR/{ids,tgt}.bin) and
// the outputs (DIR/out-G<G>[-clock]/: losses.bin, then p, m, v and the last step's g in slot order, and run.txt) are fgpt_bed.py's.
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "kvm_fused_gpt.h"

#ifndef BED_NAME
#error "a bed defines BED_NAME, the prefix of its messages, before including bed_common.cuh"
#endif

#define CK(x)                                                                                                       \
  do {                                                                                                              \
    cudaError_t e_ = (x);                                                                                           \
    if (e_ != cudaSuccess) { fprintf(stderr, BED_NAME ": CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, #x); exit(3); } \
  } while (0)

static void die(const char* fmt, const std::string& a = "") { fprintf(stderr, BED_NAME ": "); fprintf(stderr, fmt, a.c_str()); fprintf(stderr, "\n"); exit(2); }

// The phase clock: nanoseconds (%globaltimer) since the previous mark, per member and mark, summed over the run in shared memory (a mark
// costs a __syncthreads and two reads, no memory round trip) and written out at the end; and each member's SM cycles over the whole run
// beside its nanoseconds, so the bed reports the clock the SMs actually ran at. MARKS is the program's mark count.
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
template <int MARKS>
struct PhaseClock {
  unsigned long long* acc;   // [G][MARKS] ns, then [G][2]: the run's ns and cycles
  unsigned long long* sacc;  // [MARKS] in shared memory
  unsigned long long last, t0;
  long long c0;
  __device__ void start() {
    for (int k = threadIdx.x; k < MARKS; k += blockDim.x) sacc[k] = 0;
    __syncthreads();
    last = t0 = gtimer(); c0 = clock64();
  }
  __device__ void operator()(int mk) {
    __syncthreads();
    if (threadIdx.x == 0) { const unsigned long long t = gtimer(); sacc[mk] += t - last; last = t; }
  }
  __device__ void stop(int G) {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long* run = acc + (size_t)G * MARKS + 2 * blockIdx.x;
      run[0] = gtimer() - t0; run[1] = (unsigned long long)(clock64() - c0);
      for (int k = 0; k < MARKS; ++k) acc[blockIdx.x * MARKS + k] = sacc[k];
    }
  }
};

// The warm-up: every SM spins for `cycles`, so a GPU that idles at a low clock (a laptop part does) reaches its running clock before the
// timed launch. The result is never used; `sink` keeps the loop from being removed.
__global__ void warm_kernel(long long cycles, int* sink) {
  const long long t0 = clock64();
  int x = threadIdx.x;
  while (clock64() - t0 < cycles) x = x * 1103515245 + 12345;
  if (x == 42 && sink) *sink = x;
}

template <class T> static std::vector<T> read_bin(const std::string& path, size_t n) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot read %s", path);
  std::vector<T> v(n);
  f.read(reinterpret_cast<char*>(v.data()), (std::streamsize)(n * sizeof(T)));
  if ((size_t)f.gcount() != n * sizeof(T)) die("%s is shorter than the meta says", path);
  if (f.peek() != EOF) die("%s is longer than the meta says", path);
  return v;
}
template <class T> static void write_bin(const std::string& path, const T* p, size_t n) {
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(p), (std::streamsize)(n * sizeof(T)));
  if (!f) die("cannot write %s", path);
}

// ---- the workload ---------------------------------------------------------------------------------------------------------------------
// meta.txt read into a descriptor's shape, hyperparameters and slot shapes (h's buffers are null until bed_upload), with the run's first
// step and the slots' total elements.
struct BedModel {
  KvmFgptDesc h{};
  int step0 = 0;
  size_t total = 0;                           // the slots' elements: one copy of the parameters
};
static BedModel bed_meta(const std::string& dir, int G) {
  std::map<std::string, double> kv;
  struct Slot { int rows, cols; float wd; };
  std::vector<Slot> slots;
  {
    std::ifstream f(dir + "/meta.txt");
    if (!f) die("cannot read %s/meta.txt", dir);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      std::string k;
      if (!(ss >> k)) continue;
      if (k == "slot") {
        int i; Slot s;
        ss >> i >> s.rows >> s.cols >> s.wd;
        if (i != (int)slots.size()) die("slots out of order in %s/meta.txt", dir);
        slots.push_back(s);
      } else { double v; ss >> v; kv[k] = v; }
    }
  }
  auto need = [&](const char* k) { auto it = kv.find(k); if (it == kv.end()) die("meta.txt lacks %s", k); return it->second; };
  BedModel m;
  KvmFgptDesc& h = m.h;
  h.N = (int)need("N"); h.T = (int)need("T"); h.B = (int)need("B"); h.d = (int)need("d"); h.H = (int)need("H"); h.hd = (int)need("hd");
  h.F = (int)need("F"); h.V = (int)need("V"); h.L = (int)need("L"); h.G = G; h.steps = (int)need("steps"); h.mm_numerics = (int)need("mm_numerics");
  h.ln_eps = (float)need("ln_eps"); h.theta = (float)need("theta"); h.scale = (float)need("scale"); h.lr = (float)need("lr");
  h.beta1 = (float)need("beta1"); h.beta2 = (float)need("beta2"); h.adam_eps = (float)need("adam_eps"); h.max_norm = (float)need("max_norm");
  h.ids_stride = (int)need("ids_stride"); h.ids_off = (int)need("ids_off"); h.tgt_stride = (int)need("tgt_stride"); h.tgt_off = (int)need("tgt_off");
  h.ids_len = (long long)need("ids_len"); h.tgt_len = (long long)need("tgt_len");
  m.step0 = (int)need("step0");
  if (h.L < 1 || h.L > KVM_FGPT_MAXL || (int)slots.size() != kvm_fgpt_nparams(h.L)) die("the meta's layers and slots disagree");
  for (int i = 0; i < (int)slots.size(); ++i) {
    h.prows[i] = slots[i].rows; h.pcols[i] = slots[i].cols; h.wd[i] = slots[i].wd;
    m.total += (size_t)slots[i].rows * slots[i].cols;
  }
  return m;
}

// The run's device buffers as a bundle's: the parameters, their moments and gradients (one buffer each, slot after slot), the streams, the
// step counter (step0) and each step's loss buffer (a device array of pointers into one buffer). Gradients and losses start as NaN, so an
// element the program never writes shows in the comparison. `extra_bytes` is the form's own device memory: the run refuses a total above
// 1 GiB (a laptop GPU's memory also drives its screen).
struct BedRun {
  float *dp = nullptr, *dm = nullptr, *dv = nullptr, *dg = nullptr, *dloss = nullptr;
  int32_t* dstep = nullptr;
};
static BedRun bed_upload(const std::string& dir, BedModel& m, size_t extra_bytes) {
  KvmFgptDesc& h = m.h;
  const size_t total = m.total;
  const size_t bytes = 4 * total * 4 + extra_bytes + (size_t)(h.ids_len + h.tgt_len) * 4;
  if (bytes > (1ull << 30)) die("the run needs more than 1 GiB of device memory");
  const auto P = read_bin<float>(dir + "/p.bin", total), M = read_bin<float>(dir + "/m.bin", total), Vv = read_bin<float>(dir + "/v.bin", total);
  const auto ids = read_bin<int32_t>(dir + "/ids.bin", (size_t)h.ids_len), tgt = read_bin<int32_t>(dir + "/tgt.bin", (size_t)h.tgt_len);
  BedRun r;
  CK(cudaMalloc(&r.dp, total * 4)); CK(cudaMalloc(&r.dm, total * 4)); CK(cudaMalloc(&r.dv, total * 4)); CK(cudaMalloc(&r.dg, total * 4));
  CK(cudaMemcpy(r.dp, P.data(), total * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(r.dm, M.data(), total * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dv, Vv.data(), total * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(r.dg, 0xff, total * 4));   // NaN: a gradient element no member wrote shows in the comparison
  size_t off = 0;
  for (int i = 0; i < kvm_fgpt_nparams(h.L); ++i) {
    h.p[i] = r.dp + off; h.m[i] = r.dm + off; h.v[i] = r.dv + off; h.g[i] = r.dg + off;
    off += (size_t)h.prows[i] * h.pcols[i];
  }
  int32_t *dids, *dtgt;
  CK(cudaMalloc(&dids, ids.size() * 4)); CK(cudaMalloc(&dtgt, tgt.size() * 4)); CK(cudaMalloc(&r.dstep, 4));
  CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dtgt, tgt.data(), tgt.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(r.dstep, &m.step0, 4, cudaMemcpyHostToDevice));
  float** dlossp;
  CK(cudaMalloc(&r.dloss, (size_t)h.steps * 2 * 4)); CK(cudaMemset(r.dloss, 0xff, (size_t)h.steps * 2 * 4));
  std::vector<float*> lp(h.steps);
  for (int s = 0; s < h.steps; ++s) lp[s] = r.dloss + 2 * s;
  CK(cudaMalloc(&dlossp, h.steps * sizeof(float*)));
  CK(cudaMemcpy(dlossp, lp.data(), h.steps * sizeof(float*), cudaMemcpyHostToDevice));
  h.ids_src = dids; h.tgt_src = dtgt; h.step = r.dstep; h.loss = dlossp;
  return r;
}

// Warms the GPU to its running clock: about 0.5 s of busy SMs, in launches of 20 ms, right before the timed run.
static void bed_warm(int sms) {
  cudaEvent_t w0, w1;
  CK(cudaEventCreate(&w0)); CK(cudaEventCreate(&w1));
  float spent = 0.f;
  for (int i = 0; i < 200 && spent < 500.f; ++i) {
    CK(cudaEventRecord(w0));
    warm_kernel<<<sms * 2, 256>>>(20000000LL, nullptr);
    CK(cudaEventRecord(w1));
    CK(cudaEventSynchronize(w1));
    float ms1 = 0.f;
    CK(cudaEventElapsedTime(&ms1, w0, w1));
    spent += ms1;
  }
  printf(BED_NAME ": warmed the GPU for %.0f ms\n", spent);
  CK(cudaEventDestroy(w0)); CK(cudaEventDestroy(w1));
}

// The phase table from a clocked run's words (PhaseClock's layout): microseconds a step of each mark, the members' mean and largest, and the
// mean's share of the step.
static void bed_phase_table(const unsigned long long* dacc, int G, int marks, int steps, const char* const* names) {
  std::vector<unsigned long long> acc((size_t)G * (marks + 2));
  CK(cudaMemcpy(acc.data(), dacc, acc.size() * 8, cudaMemcpyDeviceToHost));
  double tot = 0.0, run_ns = 0.0, run_cyc = 0.0;
  std::vector<double> mean(marks), mx(marks);
  for (int k = 0; k < marks; ++k) {
    for (int b = 0; b < G; ++b) { const double c = (double)acc[(size_t)b * marks + k] / steps / 1000.0; mean[k] += c / G; mx[k] = std::max(mx[k], c); }
    tot += mean[k];
  }
  for (int b = 0; b < G; ++b) { run_ns += (double)acc[(size_t)G * marks + 2 * b] / G; run_cyc += (double)acc[(size_t)G * marks + 2 * b + 1] / G; }
  printf(BED_NAME ": phase clock (globaltimer), %.1f us a step in all; the SMs ran at %.0f MHz (cycles over nanoseconds, members' mean)\n", tot, 1000.0 * run_cyc / run_ns);
  printf("  %-36s %10s %10s %7s\n", "phase", "mean us", "largest", "share");
  for (int k = 0; k < marks; ++k)
    printf("  %-36s %10.2f %10.2f %6.1f%%\n", names[k], mean[k], mx[k], 100.0 * mean[k] / tot);
}

// The outputs fgpt_bed.py reads, into DIR/out-G<G>[-clock]/.
static void bed_write(const std::string& dir, const BedModel& m, const BedRun& r, bool clocked, int err, int done, int step, float ms) {
  const KvmFgptDesc& h = m.h;
  const size_t total = m.total;
  const std::string out = dir + "/out-G" + std::to_string(h.G) + (clocked ? "-clock" : "");
  mkdir(out.c_str(), 0755);
  std::vector<float> buf(total);
  CK(cudaMemcpy(buf.data(), r.dp, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/p.bin", buf.data(), total);
  CK(cudaMemcpy(buf.data(), r.dm, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/m.bin", buf.data(), total);
  CK(cudaMemcpy(buf.data(), r.dv, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/v.bin", buf.data(), total);
  CK(cudaMemcpy(buf.data(), r.dg, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/g.bin", buf.data(), total);
  std::vector<float> loss((size_t)h.steps * 2);
  CK(cudaMemcpy(loss.data(), r.dloss, loss.size() * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/losses.bin", loss.data(), loss.size());
  std::ofstream f(out + "/run.txt");
  f << "error " << err << "\nfinished " << done << "\nG " << h.G << "\nstep " << step << "\nms " << ms << "\n";
}
