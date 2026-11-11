// The fused GPT run's own kernel (the executor's fused-GPT path, KvmExec::run_fgpt): one block per gang member, every step of the run
// (kvm_fused_gpt.cuh). A kernel of its own, not a function inside the static kernel, because ptxas allocates registers per kernel: inside
// the static kernel the program got 153 registers against the 240 it gets alone, and its held values moved the other fused functions'
// allocation (the sm_120 census of Nov 11). Alone, it compiles as its bed does, and the static kernel compiles as if it did not exist.
#pragma once
#include "kvm_fused_gpt.cuh"

extern __shared__ __align__(128) unsigned char kvm_gpt_smem[];   // the dynamic arena (the same declaration as kvm_static.cuh's)

// desc == nullptr is the warm-up launch: every block returns at once, so the kernel's code is loaded outside the timed window. A fault is
// reported as the static path reports it (KVM_ERR_BOUNDS; the slot is the member's task, then the program's code and the member), and it
// sets the error word the gang's barriers read, so every member leaves at its next crossing.
__global__ void __launch_bounds__(KVM_FGPT_NT, 1) kvm_fgpt_kernel(const KvmFgptDesc* desc, KvmCounters* cnt, KvmHostErr* herr, int slot0) {
  if (!desc) return;
  unsigned dyn;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
  const int member = blockIdx.x;
  kvm_fgpt::run(*desc, member, reinterpret_cast<float*>(kvm_gpt_smem), (int)(dyn / sizeof(float)), &cnt->err_code,
                [=](int code) { kvm_fail(cnt, herr, KVM_ERR_BOUNDS, slot0 + member, code, member); });
}
