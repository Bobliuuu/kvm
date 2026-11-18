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
#include "kvm_fgpt1.cuh"

static const char* const kMarks[kvm_fgpt::MK_COUNT] = {
    "input + embedding", "LN1", "  wait: all-gather h", "  item: stage h rows, W", "  item: q k v GEMM", "  item: RoPE + keep q k v",
    "  wait: all-gather q k v", "  item: stage + scores", "  item: softmax", "  item: P V", "  item: proj partial", "attention, rest",
    "  wait: reduce-scatter proj", "proj sum + LN2", "  wait: all-gather h2", "fc1 + GELU + fc2", "  wait: reduce-scatter fc2", "fc2 sum",
    "final LN", "  wait: all-gather hf", "logits + CE stats", "  wait: CE exchange", "lse, loss", "  wait: all-gather lse", "head backward", "  wait: reduce-scatter dhf",
    "dhf sum + LNf backward", "  wait: all-gather gx (fc2)", "MLP backward", "  wait: reduce-scatter dh2", "dh2 sum + LN2 backward",
    "  wait: all-gather gx (proj)", "  item: stage gx, Wo, o, q, k, v", "  item: dWo, do, delta", "  item: q side: dWq, q's dh rows",
    "  item: P", "  item: dV, dS", "  item: dQ, dK, partials kept", "  item: RoPE back, dbq", "  doer: dW2, db1 (MLP); dWo (attention)", "  wait: dK dV partials",
    "  item: key side: sums, RoPE, dbk dbv", "  item: dWk dWv, k v dh rows", "  doer: dW1 (MLP); dWq, dbq (attention)", "attention backward, rest", "  wait: reduce-scatter dh",
    "dh sum + LN1 backward", "  wait: all-gather gx (emb)", "embedding gradient", "attention + replicated sums, squares", "norm partial",
    "  wait: norm", "AdamW", "  wait: step end"};
// The one-block form (form=1) marks its backward finer, in slots the gang's own marks name for its waits and items, in the enum's order, so
// its table reads in step order under these names.
static const char* mark_name1(int k) {
  using namespace kvm_fgpt;
  switch (k) {
    case MK_X_GFC2: return "MLP bwd: db2, dW2 | dU";
    case MK_MLPB: return "MLP bwd: db1, LN2 again";
    case MK_X_DH2: return "MLP bwd: dW1 | dh2";
    case MK_LN2B: return "MLP bwd: LN2 backward";
    case MK_UB_STAGE: return "attn bwd: dbo, dWo | dO | P";
    case MK_UB_DO: return "attn bwd: dV | dP";
    case MK_UB_STAGE2: return "attn bwd: delta, dS";
    case MK_UB_P: return "attn bwd: dQ | dK";
    case MK_UB_DV_DS: return "attn bwd: RoPE back, LN1 again";
    case MK_UB_DQ_DK: return "attn bwd: dbqkv, dWqkv | dh1";
    case MK_ATTNB: return "attn bwd: LN1 backward";
    default: return kMarks[k];
  }
}

// `acc` (plain run): member 0's in-kernel nanoseconds, start to end, so the run's time excludes the launch.
template <bool CLOCK, bool COUNTED, bool Z>   // COUNTED: the gang's crossing form (kvm_gang::counted_form(G)); Z: ZBW's instance (kvm_fgpt_zbw_of); both picked at launch
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
    r = kvm_fgpt::run<COUNTED, Z>(*d, blockIdx.x, sm, sf, err, fail, ph);
    ph.stop(d->G);
  } else {
    __shared__ unsigned long long t0;   // the run's start in shared memory, not two 64-bit registers held across the run (+0.6% a step)
    __shared__ long long c0;
    if (threadIdx.x == 0) { t0 = gtimer(); c0 = clock64(); }
    r = kvm_fgpt::run<COUNTED, Z>(*d, blockIdx.x, sm, sf, err, fail);
    if (threadIdx.x == 0 && blockIdx.x == 0 && acc) { acc[0] = gtimer() - t0; acc[1] = (unsigned long long)(clock64() - c0); }
  }
  if (threadIdx.x == 0) ok[blockIdx.x] = r ? 1 : 0;
}

// form=1 (kvm_fgpt1.cuh): the whole step in one block, its state on chip. Same arguments and outputs as the gang's kernel, one block.
template <bool CLOCK>
__global__ void __launch_bounds__(KVM_FGPT_NT, 1) fgpt1_kernel(const KvmFgptDesc* d, int smem_floats, int* err, int* ok, unsigned long long* acc,
                                                               int fault_member, int fault_floats) {
  extern __shared__ float sm[];
  (void)fault_member; (void)fault_floats;
  const auto fail = [err](int code) { atomicCAS(err, 0, code); };
  bool r;
  if constexpr (CLOCK) {
    __shared__ unsigned long long sacc[kvm_fgpt::MK_COUNT];
    PhaseClock<kvm_fgpt::MK_COUNT> ph{acc, sacc, 0, 0, 0};
    ph.start();
    r = kvm_fgpt1::run1(*d, sm, smem_floats, fail, ph);
    ph.stop(1);
  } else {
    __shared__ unsigned long long t0;   // the run's start in shared memory, not two 64-bit registers held across the run (+0.6% a step)
    __shared__ long long c0;
    if (threadIdx.x == 0) { t0 = gtimer(); c0 = clock64(); }
    auto ph = [](int) {};
    r = kvm_fgpt1::run1(*d, sm, smem_floats, fail, ph);
    if (threadIdx.x == 0 && acc) { acc[0] = gtimer() - t0; acc[1] = (unsigned long long)(clock64() - c0); }
  }
  if (threadIdx.x == 0) ok[0] = r ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: fgpt_bed DIR G [clock] [qb=Q]\n"); return 2; }
  const std::string dir = argv[1];
  const int G = atoi(argv[2]);
  bool clocked = false;
  int qb_arg = 0, rows_arg = 0, fault_member = -1, fault_floats = 0, form = 0;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "clock")) clocked = true;
    else if (!strncmp(argv[i], "qb=", 3)) qb_arg = atoi(argv[i] + 3);
    else if (!strncmp(argv[i], "rows=", 5)) rows_arg = atoi(argv[i] + 5);
    else if (!strncmp(argv[i], "form=", 5)) form = atoi(argv[i] + 5);
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
  // the instance by crossing form and ZBW (picked below, once the shared memory is known); the budget is the smaller of the two ZBW forms'
  const auto pick = [clocked, counted](bool z) -> void* {
    if (clocked) return counted ? (z ? (void*)fgpt_kernel<true, true, true> : (void*)fgpt_kernel<true, true, false>)
                                : (z ? (void*)fgpt_kernel<true, false, true> : (void*)fgpt_kernel<true, false, false>);
    return counted ? (z ? (void*)fgpt_kernel<false, true, true> : (void*)fgpt_kernel<false, true, false>)
                   : (z ? (void*)fgpt_kernel<false, false, true> : (void*)fgpt_kernel<false, false, false>);
  };
  void* kern = pick(false);
  long long budget = 0;
  {
    int dev0 = 0, optin = 0;
    CK(cudaGetDevice(&dev0));
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev0));
    cudaFuncAttributes fa, fz; CK(cudaFuncGetAttributes(&fa, pick(false))); CK(cudaFuncGetAttributes(&fz, pick(true)));
    budget = ((long long)optin - (long long)(fa.sharedSizeBytes > fz.sharedSizeBytes ? fa.sharedSizeBytes : fz.sharedSizeBytes)) / 4;
  }
  if (form != 0 && (form != 1 || G != 1)) die("form=1 runs the step in one block: G must be 1");
  if (form == 1) {   // the one-block form's own layout; the gang's row chunk and fit don't apply
    const kvm_fgpt1::Layout y1 = kvm_fgpt1::kvm_fgpt1_layout(h.N, h.T, h.d, h.H, h.F, h.V, h.L);
    int dev1 = 0, optin1 = 0;
    CK(cudaGetDevice(&dev1));
    CK(cudaDeviceGetAttribute(&optin1, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev1));
    cudaFuncAttributes f1; CK(cudaFuncGetAttributes(&f1, clocked ? (void*)fgpt1_kernel<true> : (void*)fgpt1_kernel<false>));
    const long long budget1 = ((long long)optin1 - (long long)f1.sharedSizeBytes) / 4;
    printf("fgpt_bed: form 1 (one block, state on chip): %lld floats of shared memory of this GPU's %lld a block; %d registers a thread, %zu B local; "
           "relaxed numerics mask %d (KVM_FGPT1_FAST: 1 softmax, 2 CE, 4 LayerNorm, 8 GELU; 0 keeps run's bits)\n",
           y1.total, budget1, f1.numRegs, (size_t)f1.localSizeBytes, kvm_fgpt1::kFast);
    if (y1.total == 0 || y1.total > budget1) { fprintf(stderr, "fgpt_bed: form 1 does not fit this shape in one block\n"); return 3; }
    budget = y1.total;
    h.rows = h.N;
  } else
  h.rows = rows_arg > 0 ? rows_arg : kvm_fgpt_rows_for(h.N, h.T, h.d, h.H, h.F, h.V, G, h.qb, budget);
  if (h.rows <= 0) { fprintf(stderr, "fgpt_bed: no row chunk (N, 128 or 64) fits a member of a gang of %d in this GPU's %lld floats a block: a larger gang may\n", G, budget); return 3; }
  if (h.rows > h.N || (h.rows < h.N && h.rows % 16)) die("rows must be N or a multiple of 16 below it");
  const long long smem_floats = form == 1 ? budget : kvm_fgpt_smem_floats(h.N, h.T, h.d, h.H, h.F, h.V, G, h.qb, h.rows);
  const size_t smem = (size_t)smem_floats * 4;
  int smem_arg = (int)smem_floats;
  if (smem_floats > budget) {
    fprintf(stderr, "fgpt_bed: a member of a gang of %d needs %lld floats of shared memory at %d-row chunks, above this GPU's %lld a block\n", G, smem_floats, h.rows, budget);
    return 3;
  }
  const KvmFgptZbwRun zplan = kvm_fgpt_zbw_of(h.N, h.T, h.d, h.H, h.F, G, h.qb, h.rows, smem_floats);   // as the executor picks it
  kern = form == 1 ? (clocked ? (void*)fgpt1_kernel<true> : (void*)fgpt1_kernel<false>) : pick(zplan.any());
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
  {   // ZBW (kvm_fused_gpt.cuh): the plan the run takes (kvm_fgpt_zbw_of), and so the instance it runs on
    const int fp = (h.F + G - 1) / G, nf = (h.F + fp - 1) / fp, W = (h.N / h.T) * h.H * h.qb;
    const KvmFgptZbw& z = zplan.mlp;
    printf("fgpt_bed: ZBW's instance: %s\n", zplan.any() ? "yes (its code, for the run it engages)" : "no (the step without ZBW's code)");
    if (z.zg) printf("fgpt_bed: ZBW: %d doers (members without an attention item) for %d owners, %d owner(s) a pass, %d of %d columns a block\n",
                     z.doers, nf, z.zg, z.zdb, h.d);
    else printf("fgpt_bed: ZBW: off (%d attention items for %d members, %d owners)\n", W, G, nf);
    if (zplan.attn) printf("fgpt_bed: ZBW for the attention: members %d-%d run the %d items' dWo, dWq and dbq\n", W, 2 * W - 1, W);
    else printf("fgpt_bed: ZBW for the attention: off\n");
  }
  unsigned long long* dacc = nullptr;
  const size_t acc_words = clocked ? (size_t)G * (kvm_fgpt::MK_COUNT + 2) : 2;   // plain: member 0's nanoseconds and SM cycles
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
  unsigned long long plain[2] = {0, 0};   // member 0's in-kernel nanoseconds and SM cycles, start to end (plain run)
  if (!clocked) CK(cudaMemcpy(plain, dacc, 16, cudaMemcpyDeviceToHost));
  printf("fgpt_bed: run %.3f ms with its launch; in the kernel %.1f us a step (member 0's globaltimer; a hint, not a chip's verdict); error word %d, "
         "%d of %d members finished, step counter %d -> %d\n", ms, plain[0] / 1000.0 / h.steps, err, done, G, step0, step);
  if (!clocked && plain[0] > 0) {   // the run's own clock: a ratio of cycles holds across the SM clock's states (1,410 or about 1,095 MHz)
    const double mhz = 1e3 * (double)plain[1] / (double)plain[0];
    printf("fgpt_bed: in the kernel %.2f kcycles a step at %.0f MHz (member 0's clock64 over its globaltimer)%s\n", plain[1] / 1000.0 / h.steps,
           mhz, mhz < 1380.0 ? "; SLOW CLOCK STATE: under 1,380 MHz, rerun before quoting this run's microseconds" : "");
  }
  if (clocked) {   // the phase table: microseconds a step, the members' mean and largest
    const char* names[kvm_fgpt::MK_COUNT];
    for (int k = 0; k < kvm_fgpt::MK_COUNT; ++k) names[k] = form == 1 ? mark_name1(k) : kMarks[k];
    bed_phase_table(dacc, G, kvm_fgpt::MK_COUNT, h.steps, names);
  }
  // ---- outputs
  bed_write(dir, model, run, clocked, err, done, step, ms);
  return err || done != G ? 1 : 0;
}
