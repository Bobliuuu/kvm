// The fused GPT step's bed: kvm_fgpt::run alone (include/kvm_fused_gpt.cuh), its G members as the blocks of one cooperative launch, on a model
// and data that fgpt_bed.py writes from a workload. It writes the per-step losses, the final parameters and AdamW moments, the last step's
// gradients and the step counter, for fgpt_bed.py to judge against the workload VM.
//
//   fgpt_bed DIR G [clock] [qb=Q] [rows=R] [fault=M:F]   (reads DIR/meta.txt, DIR/{p,m,v}.bin, DIR/{ids,tgt}.bin; writes DIR/out-G<G>/, or DIR/out-G<G>-clock/)
//
// qb=Q sets attention's query blocks a unit (default: kvm_fgpt_qblocks). rows=R sets the N-wide phases' row chunk (default: the largest of N,
// 128 and 64 that fits this GPU's shared memory, kvm_fgpt_rows_for). fault=M:F gives member M only F floats of shared memory, so it fails
// the first phase that needs more (KVM_FGPT_E_SCRATCH) while the others go on: a test of the gang's abort, which must end every member at its
// next crossing (the run reports error word 5 and no member finished), not hang.
//
// `clock` runs the program with its phase marks: every phase and every barrier's wait ends in a __syncthreads and a clock64() read by
// thread 0 of each member, summed over steps and layers, and the bed prints the cycles a step of each, the members' mean and their
// largest. The added syncs perturb the run slightly; its outputs must still match the VM.
//
// The run's device memory is the model's four copies of its parameters, the scratch and the streams: a few MB on the panel's GPT rows. The
// bed refuses a run above 1 GiB of device memory (a laptop GPU's memory also drives its screen).
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "kvm_fused_gpt.cuh"

#define CK(x)                                                                                                       \
  do {                                                                                                              \
    cudaError_t e_ = (x);                                                                                           \
    if (e_ != cudaSuccess) { fprintf(stderr, "fgpt_bed: CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), __FILE__, __LINE__, #x); exit(3); } \
  } while (0)

static void die(const char* fmt, const std::string& a = "") { fprintf(stderr, "fgpt_bed: "); fprintf(stderr, fmt, a.c_str()); fprintf(stderr, "\n"); exit(2); }

// The phase clock: nanoseconds (%globaltimer) since the previous mark, per member and mark, summed over the run in shared memory (a mark
// costs a __syncthreads and two reads, no memory round trip) and written out at the end; and each member's SM cycles over the whole run
// beside its nanoseconds, so the bed reports the clock the SMs actually ran at.
__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
struct PhaseClock {
  unsigned long long* acc;   // [G][MK_COUNT] ns, then [G][2]: the run's ns and cycles
  unsigned long long* sacc;  // [MK_COUNT] in shared memory
  unsigned long long last, t0;
  long long c0;
  __device__ void start() {
    for (int k = threadIdx.x; k < kvm_fgpt::MK_COUNT; k += blockDim.x) sacc[k] = 0;
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
      unsigned long long* run = acc + (size_t)G * kvm_fgpt::MK_COUNT + 2 * blockIdx.x;
      run[0] = gtimer() - t0; run[1] = (unsigned long long)(clock64() - c0);
      for (int k = 0; k < kvm_fgpt::MK_COUNT; ++k) acc[blockIdx.x * kvm_fgpt::MK_COUNT + k] = sacc[k];
    }
  }
};
static const char* const kMarks[kvm_fgpt::MK_COUNT] = {
    "input + embedding", "LN1", "  wait: all-gather h", "  item: stage h rows, W", "  item: q k v GEMM", "  item: RoPE + keep q k v",
    "  wait: all-gather q k v", "  item: stage + scores", "  item: softmax", "  item: P V", "  item: proj partial", "attention, rest",
    "  wait: reduce-scatter proj", "proj sum + LN2", "  wait: all-gather h2", "fc1 + GELU + fc2", "  wait: reduce-scatter fc2", "fc2 sum",
    "final LN", "  wait: all-gather hf", "logits + CE stats", "  wait: CE exchange", "lse, loss", "head backward", "  wait: reduce-scatter dhf",
    "dhf sum + LNf backward", "  wait: all-gather gx (fc2)", "MLP backward", "  wait: reduce-scatter dh2", "dh2 sum + LN2 backward",
    "  wait: all-gather gx (proj)", "  item: stage gx, Wo, o, q, k, v", "  item: dWo, do, delta", "  item: q side: dWq, q's dh rows",
    "  item: P", "  item: dV, dS", "  item: dQ, dK, partials kept", "  item: RoPE back, dbq", "  wait: dK dV partials",
    "  item: key side: sums, RoPE, dbk dbv", "  item: dWk dWv, k v dh rows", "attention backward, rest", "  wait: reduce-scatter dh",
    "dh sum + LN1 backward", "  wait: all-gather gx (emb)", "embedding gradient", "attention + replicated sums, squares", "norm partial",
    "  wait: norm", "AdamW", "  wait: step end"};

// `acc` (plain run): member 0's in-kernel nanoseconds, start to end, so the run's time excludes the launch.
template <bool CLOCK>
__global__ void __launch_bounds__(KVM_FGPT_NT, 1) fgpt_kernel(const KvmFgptDesc* d, int smem_floats, int* err, int* ok, unsigned long long* acc,
                                                              int fault_member, int fault_floats) {
  extern __shared__ float sm[];
  bool r;
  const int sf = (int)blockIdx.x == fault_member ? fault_floats : smem_floats;   // fault=M:F (a test of the abort): member M's smaller arena
  // the bed's report: the first code wins, in the word the barrier reads (the executor's kvm_fail does the same with its own codes)
  const auto fail = [err](int code) { atomicCAS(err, 0, code); };
  if constexpr (CLOCK) {
    __shared__ unsigned long long sacc[kvm_fgpt::MK_COUNT];
    PhaseClock ph{acc, sacc, 0, 0, 0};
    ph.start();
    r = kvm_fgpt::run(*d, blockIdx.x, sm, sf, err, fail, ph);
    ph.stop(d->G);
  } else {
    const unsigned long long t0 = gtimer();
    r = kvm_fgpt::run(*d, blockIdx.x, sm, sf, err, fail);
    if (threadIdx.x == 0 && blockIdx.x == 0 && acc) acc[0] = gtimer() - t0;
  }
  if (threadIdx.x == 0) ok[blockIdx.x] = r ? 1 : 0;
}

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

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: fgpt_bed DIR G [clock] [qb=Q]\n"); return 2; }
  const std::string dir = argv[1];
  const int G = atoi(argv[2]);
  bool clocked = false;
  int qb_arg = 0, rows_arg = 0, fault_member = -1, fault_floats = 0;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "clock")) clocked = true;
    else if (!strncmp(argv[i], "qb=", 3)) qb_arg = atoi(argv[i] + 3);
    else if (!strncmp(argv[i], "rows=", 5)) rows_arg = atoi(argv[i] + 5);
    else if (!strncmp(argv[i], "fault=", 6)) { if (sscanf(argv[i] + 6, "%d:%d", &fault_member, &fault_floats) != 2) die("fault=M:F, a member and its floats"); }
    else { fprintf(stderr, "fgpt_bed: unknown argument %s\n", argv[i]); return 2; }
  }
  // ---- meta.txt: `key value` lines; `slot i rows cols wd` a parameter
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
  KvmFgptDesc h{};
  h.N = (int)need("N"); h.T = (int)need("T"); h.B = (int)need("B"); h.d = (int)need("d"); h.H = (int)need("H"); h.hd = (int)need("hd");
  h.F = (int)need("F"); h.V = (int)need("V"); h.L = (int)need("L"); h.G = G; h.steps = (int)need("steps"); h.mm_numerics = (int)need("mm_numerics");
  h.ln_eps = (float)need("ln_eps"); h.theta = (float)need("theta"); h.scale = (float)need("scale"); h.lr = (float)need("lr");
  h.beta1 = (float)need("beta1"); h.beta2 = (float)need("beta2"); h.adam_eps = (float)need("adam_eps"); h.max_norm = (float)need("max_norm");
  h.ids_stride = (int)need("ids_stride"); h.ids_off = (int)need("ids_off"); h.tgt_stride = (int)need("tgt_stride"); h.tgt_off = (int)need("tgt_off");
  h.ids_len = (long long)need("ids_len"); h.tgt_len = (long long)need("tgt_len");
  const int step0 = (int)need("step0");
  const int np = kvm_fgpt_nparams(h.L);
  if (G < 1 || G > h.N || G > KVM_FGPT_MAX_GANG) die("G must be in [1, min(N, KVM_FGPT_MAX_GANG)]");   // the device code sizes its gang reads by the cap
  if (h.L < 1 || h.L > KVM_FGPT_MAXL || (int)slots.size() != np) die("the meta's layers and slots disagree");
  // ---- the parameters, their moments, gradients; the streams; the scratch; the gang's control words
  size_t total = 0;
  for (auto& s : slots) total += (size_t)s.rows * s.cols;
  h.qb = qb_arg > 0 ? qb_arg : kvm_fgpt_qblocks(h.B, h.H, h.T, G);
  if (h.qb < 1 || h.T % h.qb || (h.qb > 1 && (h.T / h.qb) % 16)) die("qb must split T into blocks of a multiple of 16 rows");
  const KvmFgptScratch sc = kvm_fgpt_scratch(h.N, h.T, h.d, h.H, h.F, h.V, h.L, G, h.qb);
  const size_t bytes = 4 * total * 4 + (size_t)sc.total * 4 + (size_t)(h.ids_len + h.tgt_len) * 4;
  if (bytes > (1ull << 30)) die("the run needs more than 1 GiB of device memory");
  const auto P = read_bin<float>(dir + "/p.bin", total), M = read_bin<float>(dir + "/m.bin", total), Vv = read_bin<float>(dir + "/v.bin", total);
  const auto ids = read_bin<int32_t>(dir + "/ids.bin", (size_t)h.ids_len), tgt = read_bin<int32_t>(dir + "/tgt.bin", (size_t)h.tgt_len);
  float *dp, *dm, *dv, *dg, *dx;
  CK(cudaMalloc(&dp, total * 4)); CK(cudaMalloc(&dm, total * 4)); CK(cudaMalloc(&dv, total * 4)); CK(cudaMalloc(&dg, total * 4));
  CK(cudaMemcpy(dp, P.data(), total * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm, M.data(), total * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, Vv.data(), total * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dg, 0xff, total * 4));   // NaN: a gradient element no member wrote shows in the comparison
  CK(cudaMalloc(&dx, (size_t)sc.total * 4));
  CK(cudaMemset(dx, 0xff, (size_t)sc.total * 4));   // NaN: a scratch value read before it is written shows too
  size_t off = 0;
  for (int i = 0; i < np; ++i) {
    h.p[i] = dp + off; h.m[i] = dm + off; h.v[i] = dv + off; h.g[i] = dg + off;
    h.prows[i] = slots[i].rows; h.pcols[i] = slots[i].cols; h.wd[i] = slots[i].wd;
    off += (size_t)slots[i].rows * slots[i].cols;
  }
  int32_t *dids, *dtgt, *dstep;
  CK(cudaMalloc(&dids, ids.size() * 4)); CK(cudaMalloc(&dtgt, tgt.size() * 4)); CK(cudaMalloc(&dstep, 4));
  CK(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dtgt, tgt.data(), tgt.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dstep, &step0, 4, cudaMemcpyHostToDevice));
  float* dloss; float** dlossp;
  CK(cudaMalloc(&dloss, (size_t)h.steps * 2 * 4)); CK(cudaMemset(dloss, 0xff, (size_t)h.steps * 2 * 4));
  std::vector<float*> lp(h.steps);
  for (int s = 0; s < h.steps; ++s) lp[s] = dloss + 2 * s;
  CK(cudaMalloc(&dlossp, h.steps * sizeof(float*)));
  CK(cudaMemcpy(dlossp, lp.data(), h.steps * sizeof(float*), cudaMemcpyHostToDevice));
  unsigned* dctl; int *derr, *dok;
  const size_t ctl_bytes = kvm_gang::line_words(KVM_FGPT_MAX_GANG) * sizeof(unsigned);
  CK(cudaMalloc(&dctl, ctl_bytes)); CK(cudaMemset(dctl, 0, ctl_bytes));   // the gang's control words (kvm_gang::cross)
  CK(cudaMalloc(&derr, 4)); CK(cudaMemset(derr, 0, 4));
  CK(cudaMalloc(&dok, G * 4)); CK(cudaMemset(dok, 0, G * 4));
  float* dbc; CK(cudaMalloc(&dbc, 2 * (size_t)h.steps * sizeof(float)));   // the bias corrections' table, filled by member 0
  h.ids_src = dids; h.tgt_src = dtgt; h.step = dstep; h.loss = dlossp; h.xch = dx; h.xch_floats = (size_t)sc.total; h.ctl = dctl; h.bc = dbc;
  // ---- the member's shared memory: the program's bound for the shape and the row chunk (the device checks every phase against it), within
  // this GPU's opt-in shared memory a block less the kernel's static share
  void* kern = clocked ? (void*)fgpt_kernel<true> : (void*)fgpt_kernel<false>;
  long long budget = 0;
  {
    int dev0 = 0, optin = 0;
    CK(cudaGetDevice(&dev0));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev0));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));
    budget = ((long long)optin - (long long)fa.sharedSizeBytes) / 4;
  }
  h.rows = rows_arg > 0 ? rows_arg : kvm_fgpt_rows_for(h.N, h.T, h.d, h.H, h.F, h.V, G, h.qb, budget);
  if (h.rows <= 0) { fprintf(stderr, "fgpt_bed: no row chunk (N, 128 or 64) fits a member of a gang of %d in this GPU's %lld floats a block: a larger gang may\n", G, budget); return 3; }
  if (h.rows > h.N || (h.rows < h.N && h.rows % 16)) die("rows must be N or a multiple of 16 below it");
  const long long smem_floats = kvm_fgpt_smem_floats(h.N, h.T, h.d, h.H, h.F, h.V, G, h.qb, h.rows);
  const size_t smem = (size_t)smem_floats * 4;
  int smem_arg = (int)smem_floats;
  if (smem_floats > budget) {
    fprintf(stderr, "fgpt_bed: a member of a gang of %d needs %lld floats of shared memory at %d-row chunks, above this GPU's %lld a block\n", G, smem_floats, h.rows, budget);
    return 3;
  }
  KvmFgptDesc* dd;
  CK(cudaMalloc(&dd, sizeof(KvmFgptDesc)));
  CK(cudaMemcpy(dd, &h, sizeof(KvmFgptDesc), cudaMemcpyHostToDevice));
  // ---- one cooperative launch: the driver refuses it unless all G blocks are resident at once, which the gang barrier needs
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int dev = 0, sms = 0, per_sm = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, KVM_FGPT_NT, smem));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("fgpt_bed: %s (sm_%d%d, %d SMs), G %d, %d steps, attention in %d query block(s) a unit, rows in chunks of %d of %d, shared memory %zu B a "
         "member, scratch %.2f MB, %d member(s) an SM at most\n", prop.name, prop.major, prop.minor, sms, G, h.steps, h.qb, h.rows, h.N, smem, sc.total * 4.0 / 1e6, per_sm);
  if ((long long)per_sm * sms < G) die("the gang does not fit the GPU at once");
  unsigned long long* dacc = nullptr;
  const size_t acc_words = clocked ? (size_t)G * (kvm_fgpt::MK_COUNT + 2) : 1;
  CK(cudaMalloc(&dacc, acc_words * 8)); CK(cudaMemset(dacc, 0, acc_words * 8));
  // ---- warm the GPU to its running clock: about 0.5 s of busy SMs, in launches of 20 ms, right before the timed run
  {
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
    printf("fgpt_bed: warmed the GPU for %.0f ms\n", spent);
    CK(cudaEventDestroy(w0)); CK(cudaEventDestroy(w1));
  }
  void* args[] = {&dd, &smem_arg, &derr, &dok, &dacc, &fault_member, &fault_floats};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  CK(cudaLaunchCooperativeKernel(kern, dim3(G), dim3(KVM_FGPT_NT), args, smem, 0));
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  int err = 0, step = 0;
  std::vector<int> ok(G);
  CK(cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ok.data(), dok, G * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&step, dstep, 4, cudaMemcpyDeviceToHost));
  int done = 0;
  for (int b = 0; b < G; ++b) done += ok[b];
  unsigned long long kern_ns = 0;
  if (!clocked) CK(cudaMemcpy(&kern_ns, dacc, 8, cudaMemcpyDeviceToHost));
  printf("fgpt_bed: run %.3f ms with its launch; in the kernel %.1f us a step (member 0's globaltimer; a hint, not a chip's verdict); error word %d, "
         "%d of %d members finished, step counter %d -> %d\n", ms, kern_ns / 1000.0 / h.steps, err, done, G, step0, step);
  if (clocked) {   // the phase table: microseconds a step, the members' mean and largest, and the mean's share of the step
    std::vector<unsigned long long> acc(acc_words);
    CK(cudaMemcpy(acc.data(), dacc, acc.size() * 8, cudaMemcpyDeviceToHost));
    double tot = 0.0, run_ns = 0.0, run_cyc = 0.0;
    std::vector<double> mean(kvm_fgpt::MK_COUNT), mx(kvm_fgpt::MK_COUNT);
    for (int k = 0; k < kvm_fgpt::MK_COUNT; ++k) {
      for (int b = 0; b < G; ++b) { const double c = (double)acc[(size_t)b * kvm_fgpt::MK_COUNT + k] / h.steps / 1000.0; mean[k] += c / G; mx[k] = std::max(mx[k], c); }
      tot += mean[k];
    }
    for (int b = 0; b < G; ++b) { run_ns += (double)acc[(size_t)G * kvm_fgpt::MK_COUNT + 2 * b] / G; run_cyc += (double)acc[(size_t)G * kvm_fgpt::MK_COUNT + 2 * b + 1] / G; }
    printf("fgpt_bed: phase clock (globaltimer), %.1f us a step in all; the SMs ran at %.0f MHz (cycles over nanoseconds, members' mean)\n", tot, 1000.0 * run_cyc / run_ns);
    printf("  %-36s %10s %10s %7s\n", "phase", "mean us", "largest", "share");
    for (int k = 0; k < kvm_fgpt::MK_COUNT; ++k)
      printf("  %-36s %10.2f %10.2f %6.1f%%\n", kMarks[k], mean[k], mx[k], 100.0 * mean[k] / tot);
  }
  // ---- outputs
  const std::string out = dir + "/out-G" + std::to_string(G) + (clocked ? "-clock" : "");
  mkdir(out.c_str(), 0755);
  std::vector<float> buf(total);
  CK(cudaMemcpy(buf.data(), dp, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/p.bin", buf.data(), total);
  CK(cudaMemcpy(buf.data(), dm, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/m.bin", buf.data(), total);
  CK(cudaMemcpy(buf.data(), dv, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/v.bin", buf.data(), total);
  CK(cudaMemcpy(buf.data(), dg, total * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/g.bin", buf.data(), total);
  std::vector<float> loss((size_t)h.steps * 2);
  CK(cudaMemcpy(loss.data(), dloss, loss.size() * 4, cudaMemcpyDeviceToHost)); write_bin(out + "/losses.bin", loss.data(), loss.size());
  {
    std::ofstream f(out + "/run.txt");
    f << "error " << err << "\nfinished " << done << "\nG " << G << "\nstep " << step << "\nms " << ms << "\n";
  }
  return err || done != G ? 1 : 0;
}
