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
#define BED_NAME "fgpt_bed"
#include "bed_common.cuh"
#include "kvm_fused_gpt.cuh"

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
template <bool CLOCK, bool COUNTED>   // COUNTED: the gang's crossing form (kvm_gang::counted_form(G), picked at launch)
__global__ void __launch_bounds__(KVM_FGPT_NT, 1) fgpt_kernel(const KvmFgptDesc* d, int smem_floats, int* err, int* ok, unsigned long long* acc,
                                                              int fault_member, int fault_floats) {
  extern __shared__ float sm[];
  bool r;
  const int sf = (int)blockIdx.x == fault_member ? fault_floats : smem_floats;   // fault=M:F (a test of the abort): member M's smaller arena
  // the bed's report: the first code wins, in the word the barrier reads (the executor's kvm_fail does the same with its own codes)
  const auto fail = [err](int code) { atomicCAS(err, 0, code); };
  if constexpr (CLOCK) {
    __shared__ unsigned long long sacc[kvm_fgpt::MK_COUNT];
    PhaseClock<kvm_fgpt::MK_COUNT> ph{acc, sacc, 0, 0, 0};
    ph.start();
    r = kvm_fgpt::run<COUNTED>(*d, blockIdx.x, sm, sf, err, fail, ph);
    ph.stop(d->G);
  } else {
    const unsigned long long t0 = gtimer();
    r = kvm_fgpt::run<COUNTED>(*d, blockIdx.x, sm, sf, err, fail);
    if (threadIdx.x == 0 && blockIdx.x == 0 && acc) acc[0] = gtimer() - t0;
  }
  if (threadIdx.x == 0) ok[blockIdx.x] = r ? 1 : 0;
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
  BedModel model = bed_meta(dir, G);
  KvmFgptDesc& h = model.h;
  const int step0 = model.step0;
  if (G < 1 || G > h.N || G > KVM_FGPT_MAX_GANG) die("G must be in [1, min(N, KVM_FGPT_MAX_GANG)]");   // the device code sizes its gang reads by the cap
  // ---- the parameters, their moments, gradients; the streams; the scratch; the gang's control words
  h.qb = qb_arg > 0 ? qb_arg : kvm_fgpt_qblocks(h.B, h.H, h.T, G);
  if (h.qb < 1 || h.T % h.qb || (h.qb > 1 && (h.T / h.qb) % 16)) die("qb must split T into blocks of a multiple of 16 rows");
  const KvmFgptScratch sc = kvm_fgpt_scratch(h.N, h.T, h.d, h.H, h.F, h.V, h.L, G, h.qb);
  const BedRun run = bed_upload(dir, model, (size_t)sc.total * 4);
  float* dx;
  CK(cudaMalloc(&dx, (size_t)sc.total * 4));
  CK(cudaMemset(dx, 0xff, (size_t)sc.total * 4));   // NaN: a scratch value read before it is written shows too
  unsigned* dctl; int *derr, *dok;
  const size_t ctl_bytes = kvm_gang::line_words(KVM_FGPT_MAX_GANG) * sizeof(unsigned);
  CK(cudaMalloc(&dctl, ctl_bytes)); CK(cudaMemset(dctl, 0, ctl_bytes));   // the gang's control words (kvm_gang::cross)
  CK(cudaMalloc(&derr, 4)); CK(cudaMemset(derr, 0, 4));
  CK(cudaMalloc(&dok, G * 4)); CK(cudaMemset(dok, 0, G * 4));
  float* dbc; CK(cudaMalloc(&dbc, 2 * (size_t)h.steps * sizeof(float)));   // the bias corrections' table, filled by member 0
  h.xch = dx; h.xch_floats = (size_t)sc.total; h.ctl = dctl; h.bc = dbc;
  // ---- the member's shared memory: the program's bound for the shape and the row chunk (the device checks every phase against it), within
  // this GPU's opt-in shared memory a block less the kernel's static share
  const bool counted = kvm_gang::counted_form(G);   // the gang's crossing form, an instance each
  void* kern = clocked ? (counted ? (void*)fgpt_kernel<true, true> : (void*)fgpt_kernel<true, false>)
                       : (counted ? (void*)fgpt_kernel<false, true> : (void*)fgpt_kernel<false, false>);
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
  bed_warm(sms);
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
  CK(cudaMemcpy(&step, run.dstep, 4, cudaMemcpyDeviceToHost));
  int done = 0;
  for (int b = 0; b < G; ++b) done += ok[b];
  unsigned long long kern_ns = 0;
  if (!clocked) CK(cudaMemcpy(&kern_ns, dacc, 8, cudaMemcpyDeviceToHost));
  printf("fgpt_bed: run %.3f ms with its launch; in the kernel %.1f us a step (member 0's globaltimer; a hint, not a chip's verdict); error word %d, "
         "%d of %d members finished, step counter %d -> %d\n", ms, kern_ns / 1000.0 / h.steps, err, done, G, step0, step);
  if (clocked) bed_phase_table(dacc, G, kvm_fgpt::MK_COUNT, h.steps, kMarks);   // the phase table: microseconds a step, the members' mean and largest
  // ---- outputs
  bed_write(dir, model, run, clocked, err, done, step, ms);
  return err || done != G ? 1 : 0;
}
