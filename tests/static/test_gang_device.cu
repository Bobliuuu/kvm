// Device test of the gang barriers (include/kvm_gang.cuh), the counter form (`barrier`), the flag-line form (`barrier_lines`) and the crossing
// that picks one by the gang's size (`cross`: at LinesMax 8, so every GPU's gangs take both forms, and at the shipped KVM_GANG_LINES_MAX): gangs of
// several sizes on disjoint blocks of one co-resident grid exchange slots through global memory for many rounds, and every member checks
// every value it reads. Each form runs twice: the gangs of a fused step's widths side by side, then one gang of every block (more than 32
// members on most GPUs, so the flag line polls several lines).
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude tests/static/test_gang_device.cu -o test_gang_device && ./test_gang_device [rounds]
//   compute-sanitizer --tool memcheck ./test_gang_device 50     (and racecheck, synccheck)
// Each round, each member:
//  * writes its slot;
//  * crosses the barrier, then reads every member's slot of its gang (L2 loads, ld.global.cg);
//  * crosses again, so no member overwrites a slot another is still reading.
// A value written before a crossing and missing after it is a barrier that let a member through too early.
// The second launch reuses the control words as the first left them (neither form needs a reset); it must pass the same checks and leave
// the words as a clean run does: the counter at 0 with the generation advanced by two a round, or every member's word at two a round.
// The third launch is the abort path: at round 7 the first member of the widest gang sets the run's error word and stops crossing. Its gang's
// other members must give up at that crossing (the barrier returns false) and the launch must end, not hang; the other gangs are not checked.
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include "kvm_gang.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
constexpr int NT = 256, SLOT = 64;   // floats a slot

__device__ __forceinline__ float val(int round, int member, int k) { return (float)((round * 31 + member * 7 + k) % 251); }
__device__ __forceinline__ float ld_cg(const float* p) { float v; asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p)); return v; }

struct Gangs { const int* first; const int* size; int n; };   // gang g: blocks [first[g], first[g] + size[g])

// FORM 0: the counter; 1: the flag lines; 2: cross<8>; 3: cross<KVM_GANG_LINES_MAX>. A gang of `members` takes the flag lines when
// members <= lines_max(FORM). The gang's control words: two for the counter form, line_words(size) otherwise, at `ctl + words * g` (a multiple of 32 words).
constexpr int lines_max(int form) { return form == 0 ? 0 : form == 1 ? 1 << 30 : form == 2 ? 8 : KVM_GANG_LINES_MAX; }
constexpr const char* form_name(int form) { return form == 0 ? "counter" : form == 1 ? "flag-line" : form == 2 ? "cross<8>" : "cross<KVM_GANG_LINES_MAX>"; }
template <int FORM> __host__ __device__ constexpr int ctl_words(int max_members) { return FORM == 0 ? 2 : kvm_gang::line_words(max_members); }

template <int FORM>
__global__ void __launch_bounds__(NT, 1) gang_kernel(Gangs gs, unsigned* ctl, int words, float* slots, int rounds, unsigned long long* errors, int* abort_word, int abort_test, int* gave_up) {
    int g = 0;
    while (g + 1 < gs.n && (int)blockIdx.x >= gs.first[g + 1]) ++g;   // this block's gang (first[] ascending)
    const int member = blockIdx.x - gs.first[g], members = gs.size[g];
    unsigned* const words_g = ctl + (size_t)words * g;
    unsigned epoch = FORM == 0 ? 0u : FORM == 1 ? kvm_gang::lines_epoch(words_g, member)
                   : FORM == 2 ? kvm_gang::cross_epoch<8>(words_g, member, members) : kvm_gang::cross_epoch(words_g, member, members);
    const int* const ab = abort_test ? abort_word : nullptr;
    const auto cross = [&]() {
        return FORM == 0 ? kvm_gang::barrier(words_g, members, ab)
             : FORM == 1 ? kvm_gang::barrier_lines(words_g, member, members, epoch, ab)
             : FORM == 2 ? kvm_gang::cross<8>(words_g, member, members, epoch, ab)
                         : kvm_gang::cross(words_g, member, members, epoch, ab);
    };
    float* mine = slots + (size_t)blockIdx.x * SLOT;
    const float* gang_slots = slots + (size_t)gs.first[g] * SLOT;
    unsigned long long bad = 0;
    const bool quitter = abort_test && g == gs.n - 1 && member == 0;
    for (int r = 0; r < rounds; ++r) {
        if (quitter && r == 7) { if (threadIdx.x == 0) atomicExch(abort_word, 1); return; }   // a member that fails: it never arrives again
        for (int k = threadIdx.x; k < SLOT; k += NT) mine[k] = val(r, member, k);
        if (!cross()) { if (threadIdx.x == 0) atomicAdd(gave_up, 1); return; }
        for (int i = threadIdx.x; i < members * SLOT; i += NT) {
            const int m = i / SLOT, k = i % SLOT;
            if (ld_cg(gang_slots + (size_t)m * SLOT + k) != val(r, m, k)) ++bad;
        }
        if (!cross()) { if (threadIdx.x == 0) atomicAdd(gave_up, 1); return; }
    }
    if (bad) atomicAdd(errors, bad);
}

// One form on one gang layout: two clean launches (the second from the words the first left), then the abort path. Returns the failures.
template <int FORM>
int run_form(const char* layout, const std::vector<int>& first, const std::vector<int>& size, int W, int rounds) {
    const int ng = (int)first.size(), maxm = *std::max_element(size.begin(), size.end()), words = ctl_words<FORM>(maxm);
    printf("%s barrier, %s: %d blocks in %d gang(s) (sizes", form_name(FORM), layout, W, ng);
    for (int s : size) printf(" %d", s);
    printf("), %d words a gang, %d rounds of two crossings\n", words, rounds);
    int *dfirst, *dsize; unsigned* ctl; float* slots; unsigned long long* err; int *abort_word, *gave_up;
    CK(cudaMalloc(&dfirst, ng * sizeof(int))); CK(cudaMalloc(&dsize, ng * sizeof(int)));
    CK(cudaMemcpy(dfirst, first.data(), ng * sizeof(int), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsize, size.data(), ng * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ctl, (size_t)words * ng * sizeof(unsigned))); CK(cudaMemset(ctl, 0, (size_t)words * ng * sizeof(unsigned)));
    CK(cudaMalloc(&slots, (size_t)W * SLOT * sizeof(float))); CK(cudaMalloc(&err, sizeof(unsigned long long)));
    CK(cudaMalloc(&abort_word, sizeof(int))); CK(cudaMalloc(&gave_up, sizeof(int)));
    CK(cudaMemset(abort_word, 0, sizeof(int))); CK(cudaMemset(gave_up, 0, sizeof(int)));
    int no = 0, fails = 0;
    Gangs gs{dfirst, dsize, ng};
    for (int launch = 0; launch < 2; ++launch) {   // the second launch starts from the control words the first left
        CK(cudaMemset(err, 0, sizeof(unsigned long long))); CK(cudaMemset(slots, 0xff, (size_t)W * SLOT * sizeof(float)));
        void* args[] = {&gs, &ctl, (void*)&words, &slots, (void*)&rounds, &err, &abort_word, &no, &gave_up};
        CK(cudaLaunchCooperativeKernel((void*)gang_kernel<FORM>, dim3(W), dim3(NT), args, 0, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long bad = 0; CK(cudaMemcpy(&bad, err, sizeof(bad), cudaMemcpyDeviceToHost));
        std::vector<unsigned> h((size_t)words * ng); CK(cudaMemcpy(h.data(), ctl, h.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        const unsigned want = 2u * rounds * (launch + 1);
        int ctl_bad = 0;
        for (int g = 0; g < ng; ++g) {
            const unsigned* w = h.data() + (size_t)words * g;
            if (size[g] > lines_max(FORM)) ctl_bad += w[0] != 0u || w[1] != want;   // the counter form's words
            else { bool ok = true; for (int m = 0; m < words; ++m) ok = ok && w[m] == (m < size[g] ? want : 0u); ctl_bad += !ok; }
        }
        printf("  launch %d: %llu wrong values read, %d gang(s) whose words are not as a clean run of %u crossings leaves them\n", launch, bad, ctl_bad, want);
        fails += (bad != 0) + (ctl_bad != 0);
    }
    {   // the abort path: fresh control words, the widest gang loses its first member at round 7
        CK(cudaMemset(ctl, 0, (size_t)words * ng * sizeof(unsigned))); CK(cudaMemset(err, 0, sizeof(unsigned long long)));
        CK(cudaMemset(abort_word, 0, sizeof(int))); CK(cudaMemset(gave_up, 0, sizeof(int)));
        int yes = 1;
        void* args[] = {&gs, &ctl, (void*)&words, &slots, (void*)&rounds, &err, &abort_word, &yes, &gave_up};
        CK(cudaLaunchCooperativeKernel((void*)gang_kernel<FORM>, dim3(W), dim3(NT), args, 0, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int gu = 0; CK(cudaMemcpy(&gu, gave_up, sizeof(int), cudaMemcpyDeviceToHost));
        const int expect = size[ng - 1] - 1;   // the widest gang's members but the one that quit
        printf("  abort launch: ended (no hang); %d member(s) gave up at a crossing (the widest gang has %d others; members of other gangs may give up too once the word is set)\n", gu, expect);
        fails += gu < expect;
    }
    CK(cudaFree(dfirst)); CK(cudaFree(dsize)); CK(cudaFree(ctl)); CK(cudaFree(slots)); CK(cudaFree(err)); CK(cudaFree(abort_word)); CK(cudaFree(gave_up));
    return fails;
}

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 2000;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int occ0 = 0, occ1 = 0;
    int occ2 = 0, occ3 = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ0, gang_kernel<0>, NT, 0)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ1, gang_kernel<1>, NT, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ2, gang_kernel<2>, NT, 0)); CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ3, gang_kernel<3>, NT, 0));
    const int W = std::min(prop.multiProcessorCount, std::min(std::min(occ0, occ1), std::min(occ2, occ3)) * prop.multiProcessorCount);
    printf("%s: %d co-resident blocks of %d threads\n", prop.name, W, NT);
    // gang sizes 1, 2, 3, 5, 8, 13, then one gang of what is left: gangs of a fused step's widths side by side; then one gang of every block
    std::vector<int> first, size;
    int at = 0;
    for (int s : {1, 2, 3, 5, 8, 13}) if (at + s <= W) { first.push_back(at); size.push_back(s); at += s; }
    if (at < W) { first.push_back(at); size.push_back(W - at); at = W; }
    const std::vector<int> one_first{0}, one_size{W};
    int fails = 0;
    fails += run_form<0>("side by side", first, size, W, rounds);
    fails += run_form<0>("one gang", one_first, one_size, W, rounds);
    fails += run_form<1>("side by side", first, size, W, rounds);
    fails += run_form<1>("one gang", one_first, one_size, W, rounds);
    fails += run_form<2>("side by side", first, size, W, rounds);
    fails += run_form<2>("one gang", one_first, one_size, W, rounds);
    fails += run_form<3>("side by side", first, size, W, rounds);
    fails += run_form<3>("one gang", one_first, one_size, W, rounds);
    printf("%s\n", fails ? "FAIL" : "PASS: in every form (counter, flag lines, cross at 8 and at KVM_GANG_LINES_MAX) every member read every slot of its round, every gang's control words are as a clean run leaves them, and an aborted gang ends");
    return fails ? 1 : 0;
}
