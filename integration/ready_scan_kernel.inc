// Experimental ready-only scheduler for the original KVM instruction tape.
// Keeps operator execution and dependency semantics identical to vm_kernel.
// This is intentionally a scanning reference scheduler, not the production queue.
__device__ __forceinline__ bool kvm_ready(const Instr& in, const int* ctr) {
  if (in.w0c >= 0 && ld_acquire(ctr + in.w0c) < in.w0v) return false;
  if (in.w1c >= 0 && ld_acquire(ctr + in.w1c) < in.w1v) return false;
  return true;
}

__global__ void __launch_bounds__(NT) ready_scan_kernel(
    const Instr* tape, int n, int* states, unsigned int* cursor,
    unsigned int* completed, int* ctr, float* loss) {
  __shared__ __align__(128) float sm[TILE * TILE];
  __shared__ int chosen;
  __shared__ int finished;
  for (;;) {
    if (threadIdx.x == 0) {
      chosen = -1;
      // Thread 0 owns the block-wide termination decision.
      finished = (atomicAdd(completed, 0u) >= (unsigned)n);
      if (!finished) {
        const unsigned int begin = atomicAdd(cursor, 1u);
        for (int k = 0; k < n; ++k) {
          const int idx = (int)((begin + (unsigned)k) % (unsigned)n);
          // 0=unclaimed, 1=running, 2=completed. Acquire via atomic read.
          if (atomicAdd(states + idx, 0) != 0) continue;
          const int step = idx / g_kp.perStep;
          Instr in = inst_at(tape, idx - step * g_kp.perStep, step);
          if (!kvm_ready(in, ctr)) continue;
          if (atomicCAS(states + idx, 0, 1) == 0) {
            chosen = idx;
            break;
          }
        }
      }
    }
    __syncthreads();
    if (finished) return;
    if (chosen < 0) {
      __nanosleep(128);
      continue;
    }
    const int step = chosen / g_kp.perStep;
    Instr in = inst_at(tape, chosen - step * g_kp.perStep, step);
    exec_instr(in, sm, loss);
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      if (in.s0 >= 0) atomicAdd(ctr + in.s0, 1);
      if (in.s1 >= 0) atomicAdd(ctr + in.s1, 1);
      // Publish state after counter increments. 'completed' is a count of
      // fully executed instructions, not merely claimed instructions.
      __threadfence();
      atomicExch(states + chosen, 2);
      atomicAdd(completed, 1u);
    }
    __syncthreads();
  }
}
