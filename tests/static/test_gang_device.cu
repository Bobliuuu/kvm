// Device test of the gang barrier (include/kvm_gang.cuh): gangs of several sizes on disjoint blocks of one co-resident grid exchange slots
// through global memory for many rounds, and every member checks every value it reads.
//   nvcc -O3 -arch=sm_80 -std=c++17 -Iinclude tests/static/test_gang_device.cu -o test_gang_device && ./test_gang_device [rounds]
//   compute-sanitizer --tool memcheck ./test_gang_device 50     (and racecheck, synccheck)
// Each round, each member:
//  * writes its slot;
//  * crosses the barrier, then reads every member's slot of its gang (L2 loads, ld.global.cg);
//  * crosses again, so no member overwrites a slot another is still reading.
// A value written before a crossing and missing after it is a barrier that let a member through too early.
// The second launch reuses the control words as the first left them (a sense-reversing barrier needs no reset); it must pass the same
// checks and leave every count at 0, with the generation advanced by two a round.
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

__global__ void __launch_bounds__(NT, 1) gang_kernel(Gangs gs, unsigned* ctl, float* slots, int rounds, unsigned long long* errors, int* abort_word, int abort_test, int* gave_up) {
    int g = 0;
    while (g + 1 < gs.n && (int)blockIdx.x >= gs.first[g + 1]) ++g;   // this block's gang (first[] ascending)
    const int member = blockIdx.x - gs.first[g], members = gs.size[g];
    float* mine = slots + (size_t)blockIdx.x * SLOT;
    const float* gang_slots = slots + (size_t)gs.first[g] * SLOT;
    unsigned long long bad = 0;
    const bool quitter = abort_test && g == gs.n - 1 && member == 0;
    for (int r = 0; r < rounds; ++r) {
        if (quitter && r == 7) { if (threadIdx.x == 0) atomicExch(abort_word, 1); return; }   // a member that fails: it never arrives again
        for (int k = threadIdx.x; k < SLOT; k += NT) mine[k] = val(r, member, k);
        if (!kvm_gang::barrier(ctl + 2 * g, members, abort_test ? abort_word : nullptr)) { if (threadIdx.x == 0) atomicAdd(gave_up, 1); return; }
        for (int i = threadIdx.x; i < members * SLOT; i += NT) {
            const int m = i / SLOT, k = i % SLOT;
            if (ld_cg(gang_slots + (size_t)m * SLOT + k) != val(r, m, k)) ++bad;
        }
        if (!kvm_gang::barrier(ctl + 2 * g, members, abort_test ? abort_word : nullptr)) { if (threadIdx.x == 0) atomicAdd(gave_up, 1); return; }
    }
    if (bad) atomicAdd(errors, bad);
}

int main(int argc, char** argv) {
    const int rounds = argc > 1 ? atoi(argv[1]) : 2000;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gang_kernel, NT, 0));
    const int W = std::min(prop.multiProcessorCount, occ * prop.multiProcessorCount);
    // gang sizes 1, 2, 3, 5, 8, 13, then one gang of what is left: gangs of a fused step's widths side by side
    std::vector<int> first, size;
    int at = 0;
    for (int s : {1, 2, 3, 5, 8, 13}) if (at + s <= W) { first.push_back(at); size.push_back(s); at += s; }
    if (at < W) { first.push_back(at); size.push_back(W - at); at = W; }
    const int ng = (int)first.size();
    printf("%s: %d blocks in %d gangs (sizes", prop.name, W, ng); for (int s : size) printf(" %d", s); printf("), %d rounds of two crossings\n", rounds);
    int *dfirst, *dsize; unsigned* ctl; float* slots; unsigned long long* err;
    CK(cudaMalloc(&dfirst, ng * sizeof(int))); CK(cudaMalloc(&dsize, ng * sizeof(int)));
    CK(cudaMemcpy(dfirst, first.data(), ng * sizeof(int), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsize, size.data(), ng * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ctl, 2 * ng * sizeof(unsigned))); CK(cudaMemset(ctl, 0, 2 * ng * sizeof(unsigned)));
    CK(cudaMalloc(&slots, (size_t)W * SLOT * sizeof(float))); CK(cudaMalloc(&err, sizeof(unsigned long long)));
    int *abort_word, *gave_up; CK(cudaMalloc(&abort_word, sizeof(int))); CK(cudaMalloc(&gave_up, sizeof(int)));
    CK(cudaMemset(abort_word, 0, sizeof(int))); CK(cudaMemset(gave_up, 0, sizeof(int)));
    int no = 0;
    int fails = 0;
    for (int launch = 0; launch < 2; ++launch) {   // the second launch starts from the control words the first left
        CK(cudaMemset(err, 0, sizeof(unsigned long long))); CK(cudaMemset(slots, 0xff, (size_t)W * SLOT * sizeof(float)));
        Gangs gs{dfirst, dsize, ng};
        void* args[] = {&gs, &ctl, &slots, (void*)&rounds, &err, &abort_word, &no, &gave_up};
        CK(cudaLaunchCooperativeKernel((void*)gang_kernel, dim3(W), dim3(NT), args, 0, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        unsigned long long bad = 0; CK(cudaMemcpy(&bad, err, sizeof(bad), cudaMemcpyDeviceToHost));
        std::vector<unsigned> h(2 * ng); CK(cudaMemcpy(h.data(), ctl, h.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        int ctl_bad = 0;
        for (int g = 0; g < ng; ++g) if (h[2 * g] != 0u || h[2 * g + 1] != 2u * rounds * (launch + 1)) ++ctl_bad;
        printf("launch %d: %llu wrong values read, %d gang(s) with a count left nonzero or a generation other than %u\n", launch, bad, ctl_bad, 2u * rounds * (launch + 1));
        fails += (bad != 0) + (ctl_bad != 0);
    }
    {   // the abort path: fresh control words, the widest gang loses its first member at round 7
        CK(cudaMemset(ctl, 0, 2 * ng * sizeof(unsigned))); CK(cudaMemset(err, 0, sizeof(unsigned long long)));
        Gangs gs{dfirst, dsize, ng};
        int yes = 1;
        void* args[] = {&gs, &ctl, &slots, (void*)&rounds, &err, &abort_word, &yes, &gave_up};
        CK(cudaLaunchCooperativeKernel((void*)gang_kernel, dim3(W), dim3(NT), args, 0, 0)); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int gu = 0; CK(cudaMemcpy(&gu, gave_up, sizeof(int), cudaMemcpyDeviceToHost));
        const int expect = size[ng - 1] - 1;   // the widest gang's members but the one that quit
        printf("abort launch: ended (no hang); %d member(s) gave up at a crossing (the widest gang has %d others; members of other gangs may give up too once the word is set)\n", gu, expect);
        fails += gu < expect;
    }
    printf("%s\n", fails ? "FAIL" : "PASS: every member read every slot of its round, every gang's control words are as a clean run leaves them, and an aborted gang ends");
    return fails ? 1 : 0;
}
