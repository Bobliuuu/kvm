// F's run kernel (the executor's fused-GPT path, MvmExec::run_fgpt, for a run whose form is F): one block per gang member, every step of
// the run (mvm_f2d.cuh). A kernel of its own, as the fused step's is (mvm_fgpt_kernel.cuh): ptxas allocates registers per kernel, and F's
// program sits at 254-255 of them.
#pragma once
#include "mvm_f2d.cuh"

extern __shared__ __align__(128) unsigned char mvm_gpt_smem[];   // the dynamic arena (the same declaration as mvm_static.cuh's)

// The descriptor is the kernel's __grid_constant__ parameter (about 7 KB at MVM_FGPT_MAXL layers, under the 32,764 bytes a launch carries
// on sm_70 and later with CUDA 12.1+), so the halves read their fields in parameter space. A descriptor with no gang (G 0) is the warm-up
// launch: every block returns at once, so the kernel's code is loaded outside the timed window. A fault is reported as the static path
// reports it (MVM_ERR_BOUNDS; the slot is the member's task, then the program's code and the member), and it sets the error word the
// gang's crossings read (s.abort), so every member leaves at its next crossing. Counted: the gang's crossing form, an instance each.
template <bool Counted>
__global__ void __launch_bounds__(MVM_F2D_NT, 1) mvm_f2d_kernel(const __grid_constant__ mvm_f2d::SDesc s, MvmCounters* cnt, MvmHostErr* herr, int slot0) {
  if (s.G == 0) return;
  unsigned dyn;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
  const int member = blockIdx.x;
  mvm_fgpt::NoPhases ph;
  mvm_f2d::run<Counted>(s, reinterpret_cast<float*>(mvm_gpt_smem), (int)(dyn / sizeof(float)),
                        [=](int code) { mvm_fail(cnt, herr, MVM_ERR_BOUNDS, slot0 + member, code, member); }, ph);
}
