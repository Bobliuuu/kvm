// The gang crossing's arrival fence, the three forms of include/kvm_gang.cuh's KVM_GANG_FENCE in one process: __threadfence() and then
// the release (0, shipped), the release alone (1), and CUTLASS's fence.acq_rel with a relaxed arrival (2). Each form's barrier_counted and
// barrier_lines are built in their own unit (fence_form.cu). Separate binaries run one after another read a laptop GPU's clock drift as
// much as the fence (Nov 11: the same counter form read 1,119-1,399 ns a crossing across runs), so every repetition here launches all six
// (form x barrier) back to back, in an order that rotates with the repetition, and each form is read paired by repetition against form 0
// of the same barrier. The control words are zeroed at every gang size: their contract holds for one gang, and a larger gang's new members
// would start from other epochs.
//   for F in 0 1 2; do nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude -DKVM_GANG_FENCE=$F -DFENCE_NS=f$F -c bench/gang/fence_form.cu -o fence_f$F.o; done
//   nvcc -O3 -std=c++17 -arch=sm_XX -Iinclude bench/gang/fence_bench.cu fence_f0.o fence_f1.o fence_f2.o -o fence_bench && ./fence_bench [K=2000] [REPS=15] [CSV]
// CSV, when given, gets every timed launch (G, repetition, its slot in the rotation, barrier, fence form, ns a crossing, errors), so the
// pairing can be re-derived from the raw.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "kvm_gang.cuh"
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define FENCE_FORM_DECL(NS) namespace NS { int fence(); void launch(bool, unsigned*, int*, int, int, const int*, unsigned long long*, int*); }
FENCE_FORM_DECL(f0)
FENCE_FORM_DECL(f1)
FENCE_FORM_DECL(f2)

struct Arm { const char* name; int fence; bool lines; };
constexpr int kArms = 6;
constexpr Arm kArm[kArms] = {{"counted f0", 0, false}, {"lines f0", 0, true}, {"counted f1", 1, false},
                             {"lines f1", 1, true},    {"counted f2", 2, false}, {"lines f2", 2, true}};
using Launch = void (*)(bool, unsigned*, int*, int, int, const int*, unsigned long long*, int*);
constexpr Launch kLaunch[3] = {f0::launch, f1::launch, f2::launch};

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);   // a killed run keeps the gang sizes it finished
  const int K = argc > 1 ? atoi(argv[1]) : 2000, REPS = argc > 2 ? atoi(argv[2]) : 15;
  FILE* const csv = argc > 3 ? fopen(argv[3], "w") : nullptr;
  if (argc > 3 && !csv) { fprintf(stderr, "cannot write %s\n", argv[3]); return 2; }
  if (csv) fprintf(csv, "G,rep,slot,barrier,fence,ns_per_crossing,errors\n");
  if (f0::fence() != 0 || f1::fence() != 1 || f2::fence() != 2) {
    fprintf(stderr, "fence_form units built with the wrong KVM_GANG_FENCE: f0 %d, f1 %d, f2 %d\n", f0::fence(), f1::fence(), f2::fence());
    return 2;
  }
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount, words = kvm_gang::line_words(sms);
  unsigned* ctl[kArms]; int *data, *errors, *abort_word; unsigned long long* ns;
  for (int a = 0; a < kArms; ++a) CK(cudaMalloc(&ctl[a], words * sizeof(unsigned)));
  CK(cudaMalloc(&data, sms * sizeof(int))); CK(cudaMalloc(&errors, sizeof(int))); CK(cudaMalloc(&ns, sizeof(unsigned long long)));
  CK(cudaMalloc(&abort_word, sizeof(int))); CK(cudaMemset(abort_word, 0, sizeof(int)));   // the run's abort word, never raised here
  const auto run = [&](int a, int G) {
    CK(cudaMemset(errors, 0, sizeof(int)));
    kLaunch[kArm[a].fence](kArm[a].lines, ctl[a], data, K, G, abort_word, ns, errors);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long h = 0; int e = 0;
    CK(cudaMemcpy(&h, ns, sizeof h, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&e, errors, sizeof e, cudaMemcpyDeviceToHost));
    return std::make_pair((double)h / (2.0 * K), e);
  };
  printf("%s (sm_%d%d, %d SMs): %d crossings a launch, %d repetitions of all six arms in a rotating order; ns a crossing\n", p.name,
         p.major, p.minor, sms, 2 * K, REPS);
  printf("f0 = __threadfence() + release (shipped), f1 = release alone, f2 = fence.acq_rel + relaxed; each paired against f0 of its barrier\n");
  printf("%4s  %-11s %7s %7s %7s   %-24s %-7s %s\n", "G", "arm", "median", "min", "max", "paired: median (min..max)", "faster", "errors");
  for (int G : {2, 4, 8, 16, 24, 32, 33, 40, 48, 56, 64, 96, 108, 128, 132}) {
    if (G > sms) continue;
    for (int a = 0; a < kArms; ++a) CK(cudaMemset(ctl[a], 0, words * sizeof(unsigned)));
    std::vector<double> t[kArms];
    int errs[kArms] = {};
    for (int a = 0; a < kArms; ++a) errs[a] += run(a, G).second;   // warm-up: loads each unit's code and the clock; not timed
    for (int r = 0; r < REPS; ++r)
      for (int i = 0; i < kArms; ++i) {
        const int a = (i + r) % kArms;
        const auto [ns_x, e] = run(a, G);
        t[a].push_back(ns_x); errs[a] += e;
        if (csv) fprintf(csv, "%d,%d,%d,%s,%d,%.3f,%d\n", G, r, i, kArm[a].lines ? "lines" : "counted", kArm[a].fence, ns_x, e);
      }
    for (int a = 0; a < kArms; ++a) {
      std::vector<double> s = t[a], d;
      std::sort(s.begin(), s.end());
      const int base = kArm[a].lines ? 1 : 0;   // f0 of the same barrier
      int faster = 0;
      for (int r = 0; r < REPS; ++r) { d.push_back(t[a][r] - t[base][r]); faster += d.back() < 0; }
      std::sort(d.begin(), d.end());
      char paired[64] = "-";
      if (a != base) snprintf(paired, sizeof paired, "%+.0f (%+.0f..%+.0f)", d[d.size() / 2], d.front(), d.back());
      char fast[16] = "-";
      if (a != base) snprintf(fast, sizeof fast, "%d/%d", faster, REPS);
      printf("%4d  %-11s %7.0f %7.0f %7.0f   %-24s %-7s %d\n", G, kArm[a].name, s[s.size() / 2], s.front(), s.back(), paired, fast, errs[a]);
    }
  }
  if (csv && fclose(csv) != 0) { fprintf(stderr, "cannot finish %s\n", argv[3]); return 2; }
  return 0;
}
