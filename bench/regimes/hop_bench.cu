// Cost-model micro benchmark: R independent chains of L dependent `add` tasks (x += 1 over n floats), VM vs ordinary CUDA.
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude --fmad=false -Xcompiler -ffp-contract=off -DKVM_EXEC_NO_MAIN [-DKVM_CHAIN] bench/regimes/hop_bench.cu -o hop_bench
//   ./hop_bench [--reps 5] [--n 256] [--cases R:L,R:L,...]
// R=1 is the dependent-hop cost (a single run's critical path); R>1 are R independent runs filling the GPU. Same math everywhere
// (out = in + 1, exact in fp32, so every final buffer must be bitwise L): that is the correctness gate, checked for every variant.
// Variants, all timed back to back in this process per rep, best of reps reported (us for the whole R x L work):
//   vm         one persistent-kernel launch, R*L tasks, chain edges (kernel_ms of the executor: admission + execution)
//   per_op     R*L ordinary kernel launches on one stream, run after run (the naive baseline)
//   graph_seq  the same R*L launches captured in ONE CUDA graph (a sequential chain: R runs one after another)
//   graph_par  one CUDA graph with R parallel branches of L dependent kernels (the graph's own concurrency)
//   batched    L launches of ONE kernel over all R runs (vmap-like), captured in a graph: needs identical runs
#include "../../kvm_exec.cu"
#include <cstdio>

__global__ void addk(const float* a, const float* b, float* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] + b[i]; }
static const int TPB = 256;
static unsigned blocks_for(int n) { return (unsigned)((n + TPB - 1) / TPB); }

struct Case { int R, L, n; };
static std::vector<float> g_ones;

// ---- VM ----
static double run_vm(const Case& c, bool* ok, bool verify) {
  KvmConfig cfg; cfg.timeout_us = 60ull * 1000000;
  KvmExec ex(cfg);
  KvmInit ones; ones.kind = KvmInit::FILL; ones.value = 1.f;
  const int one = ex.add_buffer(1, c.n, ones);
  std::vector<int> X(c.R), Y(c.R);
  for (int r = 0; r < c.R; ++r) { X[r] = ex.add_buffer(1, c.n); Y[r] = ex.add_buffer(1, c.n); }
  KvmBundleBuilder bb; std::vector<int> prev(c.R, -1);
  for (int l = 0; l < c.L; ++l)
    for (int r = 0; r < c.R; ++r) {
      const int in = (l % 2 == 0) ? X[r] : Y[r], out = (l % 2 == 0) ? Y[r] : X[r];
      const int t = bb.add("add", {KvmArg::buf(in), KvmArg::buf(one), KvmArg::buf(out), KvmArg::i32(c.n)});
      if (prev[r] >= 0) bb.edge(prev[r], t);
      prev[r] = t;
    }
  KvmBundle b = bb.build();
  ex.start();
  KvmSubmit st = (int)b.tasks.size() > cfg.live_cap || b.words() > (size_t)cfg.ring_words / 2 ? ex.submit_stream(b) : ex.submit(b);
  if (st != KvmSubmit::OK) { fprintf(stderr, "vm submit failed: %s\n", kvm_submit_name(st)); *ok = false; ex.close(); ex.finish(); return 0; }
  const KvmResult& res = ex.finish();
  if (!res.ok()) { fprintf(stderr, "vm device error %s\n", kvm_err_name(res.err_code)); *ok = false; return 0; }
  if (verify)
    for (int r = 0; r < c.R; ++r) {
      std::vector<float> v = ex.read_buffer(c.L % 2 ? Y[r] : X[r]);
      for (float f : v) if (f != (float)c.L) { fprintf(stderr, "vm: run %d value %g != %d\n", r, f, c.L); *ok = false; return 0; }
    }
  return res.kernel_ms * 1e3;
}

// ---- ordinary CUDA ----
struct Dev {
  Case c; float *one, *X, *Y;   // X, Y: R*n each
  explicit Dev(const Case& cc) : c(cc) {
    CK(cudaMalloc(&one, (size_t)c.R * c.n * 4)); CK(cudaMalloc(&X, (size_t)c.R * c.n * 4)); CK(cudaMalloc(&Y, (size_t)c.R * c.n * 4));
    g_ones.assign((size_t)c.R * c.n, 1.f); CK(cudaMemcpy(one, g_ones.data(), (size_t)c.R * c.n * 4, cudaMemcpyHostToDevice));
  }
  ~Dev() { cudaFree(one); cudaFree(X); cudaFree(Y); }
  void reset() { CK(cudaMemset(X, 0, (size_t)c.R * c.n * 4)); CK(cudaMemset(Y, 0, (size_t)c.R * c.n * 4)); }
  void launch(cudaStream_t s, int r, int l) {
    const float* in = ((l % 2 == 0) ? X : Y) + (size_t)r * c.n; float* out = ((l % 2 == 0) ? Y : X) + (size_t)r * c.n;
    addk<<<blocks_for(c.n), TPB, 0, s>>>(in, one + (size_t)r * c.n, out, c.n);
  }
  void launch_batched(cudaStream_t s, int l) {
    const float* in = (l % 2 == 0) ? X : Y; float* out = (l % 2 == 0) ? Y : X;
    addk<<<blocks_for(c.R * c.n), TPB, 0, s>>>(in, one, out, c.R * c.n);
  }
  bool check() {
    std::vector<float> v((size_t)c.R * c.n); CK(cudaMemcpy(v.data(), c.L % 2 ? Y : X, v.size() * 4, cudaMemcpyDeviceToHost));
    for (float f : v) if (f != (float)c.L) { fprintf(stderr, "cuda: value %g != %d\n", f, c.L); return false; }
    return true;
  }
};
static double time_stream(cudaStream_t s, const std::function<void()>& fn) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a, s)); fn(); CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b)); cudaEventDestroy(a); cudaEventDestroy(b); return ms * 1e3;
}
static cudaGraphExec_t capture(cudaStream_t s, const std::function<void()>& body) {
  cudaGraph_t g; cudaGraphExec_t e;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); body(); CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&e, g, 0)); cudaGraphDestroy(g); return e;
}

int main(int argc, char** argv) {
  int reps = 5, n = 256; std::string cases = "1:1000,1:100,4:250,16:250,64:100,108:100,256:100";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]); else if (a == "--n" && i + 1 < argc) n = atoi(argv[++i]); else if (a == "--cases" && i + 1 < argc) cases = argv[++i];
    else { fprintf(stderr, "usage: hop_bench [--reps N] [--n floats] [--cases R:L,...]\n"); return 2; }
  }
  try {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
#ifdef KVM_CHAIN
    const char* flavour = "KVM_CHAIN";
#else
    const char* flavour = "base";
#endif
    printf("# hop_bench: %s, %d SMs, executor %s, n=%d floats per run, best of %d\n", p.name, p.multiProcessorCount, flavour, n, reps);
    printf("# %-9s %5s %5s | %10s %10s %10s %10s %10s | us per dependent step (whole set / L)\n", "", "R", "L", "vm", "per_op", "graph_seq", "graph_par", "batched");
    cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    std::stringstream ss(cases); std::string tok;
    while (std::getline(ss, tok, ',')) {
      Case c{0, 0, n}; if (sscanf(tok.c_str(), "%d:%d", &c.R, &c.L) != 2 || c.R < 1 || c.L < 1) { fprintf(stderr, "bad case %s\n", tok.c_str()); return 2; }
      Dev d(c); bool ok = true;
      // graphs
      cudaGraphExec_t g_seq = capture(s, [&] { for (int r = 0; r < c.R; ++r) for (int l = 0; l < c.L; ++l) d.launch(s, r, l); });
      std::vector<cudaStream_t> fs(c.R); for (auto& f : fs) CK(cudaStreamCreateWithFlags(&f, cudaStreamNonBlocking));
      cudaEvent_t fork, join; CK(cudaEventCreate(&fork)); std::vector<cudaEvent_t> joins(c.R); for (auto& e : joins) CK(cudaEventCreate(&e));
      cudaGraphExec_t g_par = capture(s, [&] {
        CK(cudaEventRecord(fork, s));
        for (int r = 0; r < c.R; ++r) { CK(cudaStreamWaitEvent(fs[r], fork)); for (int l = 0; l < c.L; ++l) d.launch(fs[r], r, l); CK(cudaEventRecord(joins[r], fs[r])); }
        for (int r = 0; r < c.R; ++r) CK(cudaStreamWaitEvent(s, joins[r]));
      });
      cudaGraphExec_t g_bat = capture(s, [&] { for (int l = 0; l < c.L; ++l) d.launch_batched(s, l); });
      double best[5] = {1e18, 1e18, 1e18, 1e18, 1e18};
      auto upd = [&](int k, double us) { best[k] = std::min(best[k], us); };
      for (int rep = -1; rep < reps; ++rep) {   // rep -1 is warm-up and the correctness gate
        const bool gate = rep < 0;
        double v = run_vm(c, &ok, gate); if (!ok) return 1; if (!gate) upd(0, v);
        d.reset(); CK(cudaDeviceSynchronize());
        double t = time_stream(s, [&] { for (int r = 0; r < c.R; ++r) for (int l = 0; l < c.L; ++l) d.launch(s, r, l); });
        if (gate && !d.check()) return 1; if (!gate) upd(1, t);
        d.reset(); CK(cudaDeviceSynchronize());   // graphs: the reset is captured for g_seq only; reset here for all
        t = time_stream(s, [&] { CK(cudaGraphLaunch(g_seq, s)); }); if (gate && !d.check()) return 1; if (!gate) upd(2, t);
        d.reset(); CK(cudaDeviceSynchronize());
        t = time_stream(s, [&] { CK(cudaGraphLaunch(g_par, s)); }); if (gate && !d.check()) return 1; if (!gate) upd(3, t);
        d.reset(); CK(cudaDeviceSynchronize());
        t = time_stream(s, [&] { CK(cudaGraphLaunch(g_bat, s)); }); if (gate && !d.check()) return 1; if (!gate) upd(4, t);
      }
      printf("  %-9s %5d %5d | %10.2f %10.2f %10.2f %10.2f %10.2f | ratio vs vm: per_op %.2fx graph_seq %.2fx graph_par %.2fx batched %.2fx\n", "", c.R, c.L,
             best[0] / c.L, best[1] / c.L, best[2] / c.L, best[3] / c.L, best[4] / c.L, best[1] / best[0], best[2] / best[0], best[3] / best[0], best[4] / best[0]);
      fflush(stdout);
      cudaGraphExecDestroy(g_seq); cudaGraphExecDestroy(g_par); cudaGraphExecDestroy(g_bat);
      for (auto f : fs) cudaStreamDestroy(f); for (auto e : joins) cudaEventDestroy(e); cudaEventDestroy(fork);
    }
    printf("GATE PASS: every final buffer of every variant was exactly L\n");
  } catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
  return 0;
}
