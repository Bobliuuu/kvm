// F's bed: kvm_f2d::run alone (include/kvm_f2d.cuh), its gang as the blocks of one cooperative launch, on a model and data that
// fgpt_bed.py writes from a workload. The descriptor comes from kvm_f2d::build, the executor's builder, over the workload's buffers laid out
// as a bundle's (bed_common.cuh), so the bed runs the executor's wiring. It writes the per-step losses, the final parameters and AdamW
// moments, the last step's gradients and the step counter, for fgpt_bed.py to judge against the workload VM.
//
//   nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude [-DKVM_F2D_PRECISE=1] [-DKVM_F2D_STAGES=2] bench/fused_gpt/f2d_bed.cu -o f2d_bed
//   f2d_bed DIR G [att=GtxGc] [mlp=GtxGc] [clock] [fault=M:F]   (reads DIR/meta.txt, DIR/{p,m,v}.bin, DIR/{ids,tgt}.bin; writes DIR/out-G<G>/)
//
// G is the gang: kvm_f2d::fit's own choice when it is 0 (each half's largest layout within this GPU's SMs), else the gang to run, with each
// half's largest layout within it unless att= or mlp= gives one (the attention's Gc a multiple of 3). The build's numerics must be the
// workload's (mm_numerics 0, or 16 with KVM_F2D_PRECISE=1). fault=M:F gives member M only F floats of shared memory, so it fails the run's
// first check (KVM_FGPT_E_SCRATCH) while the others go on: a test of the gang's abort, which must end every member at its first crossing
// (the run reports error word 5 and no member finished), not hang. `clock` runs the program with a phase clock at every mark and prints
// the table (microseconds a step, the members' mean and largest).
#define BED_NAME "f2d_bed"
#include "bed_common.cuh"
#include "kvm_f2d.cuh"

// the phase table's rows, in mark order (kvm_f2d.cuh's APh, MPh, SPh)
static const char* const kMarks[] = {
    "LN1 (own rows)", "  wait: h gathered", "q|k|v (stream H, Wqkv cols) + bias", "  wait: q|k|v gathered",
    "attention items (RoPE, scores, softmax, P V)", "  wait: o gathered", "proj partial (o tile resident, Wo rows)",
    "  wait: proj partials", "proj reduce-scatter + bo + x", "  wait: forward end", "d_o = dY Wo^T (stream dY, Wo rows)",
    "dWo partial (o tile resident), dbo", "  wait: d_o gathered", "items backward, query blocks", "  wait: dK dV partials",
    "items backward, key blocks", "  wait: dq|dk|dv gathered", "dWqkv, dbqkv, dH partials (dqkv tile resident)", "  wait: dH partials",
    "dH reduce-scatter, LN1 backward (own rows)", "  wait: LN1 partials", "LN2 (own rows)", "  wait: h2 gathered",
    "fc1 (stream H2, W1 cols) + GELU", "fc2 partial (W2 rows)", "  wait: fc2 partials", "fc2 reduce-scatter + b2 + residual",
    "  wait: xo gathered", "MLP: A restaged, dA = dxo W2^T + GELU'", "MLP: dW2 partial (stream dxo)", "MLP: dX partial (W1 rows)",
    "MLP: dW1 partial (stream H2)", "  wait: MLP backward partials", "dh2 reduce-scatter + LN2 backward", "  wait: dxm gathered",
    "embedding (own rows)", "  wait: x0 gathered", "final LayerNorm (own rows)", "  wait: hf gathered",
    "head: logits and softmax statistics (shard)", "  wait: statistics gathered", "loss", "head backward: dhead, dhf partials (shard)",
    "  wait: dhf partials", "dhf reduce-scatter + LNf backward", "  wait: dx[L] gathered", "embedding backward (shard)",
    "  wait: dx[0] read", "dW sums, every layer (flat shares)", "  wait: gradients summed", "squares (flat shares)",
    "  wait: the norm's partials", "AdamW (flat shares)", "  wait: step end"};
static_assert(sizeof(kMarks) / sizeof(kMarks[0]) == kvm_f2d::S_N, "a name for every phase mark");

template <bool CLOCK, bool COUNTED>   // COUNTED: the gang's crossing form (kvm_gang::counted_form(G), picked at launch)
__global__ void __launch_bounds__(KVM_F2D_NT, 1) f2d_kernel(const __grid_constant__ kvm_f2d::SDesc s, int smem_floats, int* err, int* ok,
                                                            unsigned long long* acc, int fault_member, int fault_floats) {
  extern __shared__ float sm[];
  bool r;
  const int sf = (int)blockIdx.x == fault_member ? fault_floats : smem_floats;   // fault=M:F (a test of the abort): member M's smaller arena
  const auto fail = [err](int code) { atomicCAS(err, 0, code); };               // the first code wins, in the word the crossings read
  if constexpr (CLOCK) {
    __shared__ unsigned long long sacc[kvm_f2d::S_N];
    PhaseClock<kvm_f2d::S_N> ph{acc, sacc, 0, 0, 0};
    ph.start();
    r = kvm_f2d::run<COUNTED>(s, sm, sf, fail, ph);
    ph.stop(s.G);
  } else {
    const unsigned long long t0 = gtimer();
    kvm_fgpt::NoPhases ph;
    r = kvm_f2d::run<COUNTED>(s, sm, sf, fail, ph);
    if (threadIdx.x == 0 && blockIdx.x == 0 && acc) acc[0] = gtimer() - t0;
  }
  if (threadIdx.x == 0) ok[blockIdx.x] = r ? 1 : 0;
}

static bool parse_layout(const char* arg, const char* key, int& gt, int& gc) {
  const size_t kl = strlen(key);
  if (strncmp(arg, key, kl) || arg[kl] != '=') return false;
  if (sscanf(arg + kl + 1, "%dx%d", &gt, &gc) != 2 || gt <= 0 || gc <= 0) die("bad %s (want KEY=GtxGc)", arg);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: f2d_bed DIR G [att=GtxGc] [mlp=GtxGc] [clock] [fault=M:F]\n"); return 2; }
  const std::string dir = argv[1];
  int G = atoi(argv[2]);
  bool clocked = false;
  int gta = 0, gca = 0, gtm = 0, gcm = 0, fault_member = -1, fault_floats = 0;
  for (int i = 3; i < argc; ++i) {
    if (parse_layout(argv[i], "att", gta, gca) || parse_layout(argv[i], "mlp", gtm, gcm)) continue;
    if (!strcmp(argv[i], "clock")) clocked = true;
    else if (!strncmp(argv[i], "fault=", 6)) { if (sscanf(argv[i] + 6, "%d:%d", &fault_member, &fault_floats) != 2) die("fault=M:F, a member and its floats"); }
    else { fprintf(stderr, "f2d_bed: unknown argument %s\n", argv[i]); return 2; }
  }
  int dev = 0, sms = 0, optin = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  const bool counted = kvm_gang::counted_form(G);
  void* kern = clocked ? (counted ? (void*)f2d_kernel<true, true> : (void*)f2d_kernel<true, false>)
                       : (counted ? (void*)f2d_kernel<false, true> : (void*)f2d_kernel<false, false>);
  long long budget = 0;
  {
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));
    budget = ((long long)optin - (long long)fa.sharedSizeBytes) / 4;
  }
  BedModel model = bed_meta(dir, G);
  KvmFgptDesc& h = model.h;
  // ---- the form: kvm_f2d::fit within the gang (or the GPU's SMs), then any layout the arguments give
  kvm_f2d::Fit f;
  if (const char* why = kvm_f2d::fit(h.N, h.T, h.B, h.d, h.H, h.F, h.V, h.L, h.mm_numerics, budget, G > 0 ? G : sms, f)) die("F cannot run this workload: %s", why);
  if (G > 0) f.G = G;
  if (gta) { f.gta = gta; f.gca = gca; }
  if (gtm) { f.gtm = gtm; f.gcm = gcm; }
  if (f.G < f.gta * f.gca || f.G < f.gtm * f.gcm) die("the gang holds neither layout in full");
  G = h.G = f.G;
  f.qb = kvm_fgpt_qblocks(h.B, h.H, h.T, G);
  f.nv = std::min(G, (h.V + 15) / 16); f.vp = (h.V + f.nv - 1) / f.nv;
  {
    kvm_f2d::ATiling ta; kvm_f2d::MTiling tm;
    if (const char* why = kvm_f2d::attn_tiling(h.N, h.d, f.gta, f.gca, f.kc, ta)) die("attention layout: %s", why);
    if (const char* why = kvm_f2d::mlp_tiling(h.N, h.d, h.F, f.gtm, f.gcm, f.kc, tm)) die("MLP layout: %s", why);
    f.smem_floats = kvm_f2d::smem_floats(ta, tm, h.T, h.T / f.qb, h.d, h.hd, f.vp, f.kc);
  }
  if (f.smem_floats > budget) { fprintf(stderr, "f2d_bed: a member needs %lld floats of shared memory, above this GPU's %lld a block\n", f.smem_floats, budget); return 3; }
  // ---- the workload's buffers, then F's own through the builder (zeroed device memory, counted against the bed's 1 GiB)
  BedRun run = bed_upload(dir, model, 0);
  size_t f2d_bytes = 0;
  std::vector<void*> allocs;
  auto alloc = [&](size_t n) {
    const size_t bytes = std::max<size_t>(n, 1) * sizeof(float);
    f2d_bytes += bytes;
    if (4 * model.total * 4 + f2d_bytes > (1ull << 30)) die("the run needs more than 1 GiB of device memory");
    float* p = nullptr;
    CK(cudaMalloc(&p, bytes)); CK(cudaMemset(p, 0, bytes));
    allocs.push_back(p);
    return p;
  };
  kvm_f2d::SDesc s;
  if (const char* why = kvm_f2d::build(h, f, alloc, s)) die("the builder refused the run: %s", why);
  int *derr, *dok;
  CK(cudaMalloc(&derr, 4)); CK(cudaMemset(derr, 0, 4));
  CK(cudaMalloc(&dok, G * 4)); CK(cudaMemset(dok, 0, G * 4));
  kvm_f2d::set_abort(s, derr);
  // ---- one cooperative launch: the driver refuses it unless all G blocks are resident at once, which the gang's crossings need
  const size_t smem = (size_t)f.smem_floats * 4;
  int smem_arg = (int)f.smem_floats, per_sm = 0;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, KVM_F2D_NT, smem));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("f2d_bed: %s (sm_%d%d, %d SMs): L %d, d %d, H %d x %d, F %d, V %d, %d x %d rows, %d step(s) from %d; gang %d (%s); attention %d x %d, "
         "MLP %d x %d, KC %d, %d query block(s) a unit, %d vocabulary shards of %d; shared %.1f KB a member, F's buffers %.1f MB; %s build\n",
         prop.name, prop.major, prop.minor, sms, h.L, h.d, h.H, h.hd, h.F, h.V, h.B, h.T, h.steps, model.step0, G, counted ? "counted" : "flag lines",
         f.gta, f.gca, f.gtm, f.gcm, f.kc, f.qb, f.nv, f.vp, smem / 1024.0, f2d_bytes / 1e6, kvm_f2d::kPrec ? "3xTF32" : "1xTF32");
  if ((long long)per_sm * sms < G) die("the gang does not fit the GPU at once");
  unsigned long long* dacc = nullptr;
  const size_t acc_words = clocked ? (size_t)G * (kvm_f2d::S_N + 2) : 1;
  CK(cudaMalloc(&dacc, acc_words * 8)); CK(cudaMemset(dacc, 0, acc_words * 8));
  bed_warm(sms);
  void* args[] = {&s, &smem_arg, &derr, &dok, &dacc, &fault_member, &fault_floats};
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  CK(cudaLaunchCooperativeKernel(kern, dim3(G), dim3(KVM_F2D_NT), args, smem, 0));
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  int err = 0, step = 0;
  std::vector<int> ok(G);
  CK(cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ok.data(), dok, G * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&step, run.dstep, 4, cudaMemcpyDeviceToHost));
  int done = 0;
  for (int b = 0; b < G; ++b) done += ok[b];
  unsigned long long kern_ns = 0;
  if (!clocked) CK(cudaMemcpy(&kern_ns, dacc, 8, cudaMemcpyDeviceToHost));
  printf("f2d_bed: run %.3f ms with its launch; in the kernel %.1f us a step (member 0's globaltimer; a hint, not a chip's verdict); error word %d, "
         "%d of %d members finished, step counter %d -> %d\n", ms, kern_ns / 1000.0 / h.steps, err, done, G, model.step0, step);
  if (clocked) bed_phase_table(dacc, G, kvm_f2d::S_N, h.steps, kMarks);
  bed_write(dir, model, run, clocked, err, done, step, ms);
  for (void* p : allocs) CK(cudaFree(p));
  return err || done != G ? 1 : 0;
}
