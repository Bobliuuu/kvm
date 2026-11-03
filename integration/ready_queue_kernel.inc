// KVM experimental dependency-driven ready queue.
// Full-barrier counter waits are lowered into a CSR DAG on the host.
// Capacity is exactly ntape; every instruction is enqueued once, so the
// queue never wraps or becomes full. No producer blocks on capacity.
#include <cuda/atomic>

__device__ __forceinline__ int kvm_q_load(int* p) {
  cuda::atomic_ref<int, cuda::thread_scope_device> a(*p);
  return a.load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void kvm_q_publish(int* p, int v) {
  cuda::atomic_ref<int, cuda::thread_scope_device> a(*p);
  a.store(v, cuda::memory_order_release);
}
__device__ __forceinline__ int kvm_q_inc(int* p) {
  cuda::atomic_ref<int, cuda::thread_scope_device> a(*p);
  return a.fetch_add(1, cuda::memory_order_acq_rel);
}
__device__ __forceinline__ int kvm_q_dec(int* p) {
  cuda::atomic_ref<int, cuda::thread_scope_device> a(*p);
  return a.fetch_sub(1, cuda::memory_order_acq_rel);
}
__device__ __forceinline__ bool kvm_q_claim(int* p, int expected) {
  cuda::atomic_ref<int, cuda::thread_scope_device> a(*p);
  return a.compare_exchange_strong(expected, expected + 1,
                                   cuda::memory_order_acq_rel);
}

__global__ void __launch_bounds__(NT) ready_queue_kernel(
    const Instr* tape, int n, const int* offsets, const int* successors,
    int* remaining, int* queue_task, int* queue_ready,
    int* head, int* tail, int* done, float* loss, int first) {
  __shared__ __align__(128) float sm[TILE * TILE];
  __shared__ int chosen, stop;
  for (;;) {
    if (threadIdx.x == 0) {
      chosen = -1;
      stop = (kvm_q_load(done) == n);
      if (!stop) {
        // Only advance the head after its slot is published. A temporarily
        // unpublished reservation may cause spinning, never skipped work.
        int h = kvm_q_load(head);
        if (h < kvm_q_load(tail) && kvm_q_load(queue_ready + h)) {
          if (kvm_q_claim(head, h)) chosen = queue_task[h];
        }
      }
    }
    __syncthreads();
    if (stop) return;
    if (chosen < 0) { __nanosleep(64); continue; }
    int global_id = chosen + first;
    int step = global_id / g_kp.perStep;
    Instr in = inst_at(tape, global_id - step * g_kp.perStep, step);
    exec_instr(in, sm, loss);
    __syncthreads();
    if (threadIdx.x == 0) {
      // The block barrier completes all operator writes before the
      // release/acq_rel dependency transitions below.
      __threadfence();
      for (int e = offsets[chosen]; e < offsets[chosen + 1]; ++e) {
        int succ = successors[e];
        if (kvm_q_dec(remaining + succ) == 1) {
          int slot = kvm_q_inc(tail);
          // Host guarantees capacity >= n, with exactly one enqueue/task.
          queue_task[slot] = succ;
          kvm_q_publish(queue_ready + slot, 1);
        }
      }
      kvm_q_inc(done);
    }
    __syncthreads();
  }
}
