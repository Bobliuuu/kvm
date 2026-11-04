#pragma once
// KVM task ABI v2: the contract between the host compiler (runtime/compile_workload.py) and the GPU executor.
// Little-endian. Every field is a 32-bit word. The device never parses JSON: arguments are typed words.
#include <cstdint>

#define KVM_ABI_VERSION 2
#define KVM_NO_VAR (-1)

// One task. Successor indices and wait/publish variable ids are batch-local for successors, global for variables.
// A batch is one compiled workload (or one admission epoch); the executor adds the batch's base slot to successors.
struct alignas(4) KvmTaskDesc {
    int32_t op_id;           // index into the operator registry (include/kvm_ops.def)
    int32_t succ_off;        // CSR range into the batch's successor array
    int32_t succ_cnt;
    int32_t indegree;        // number of in-batch predecessors still to finish
    int32_t priority;        // larger runs first among ready tasks
    int32_t deadline_us;     // -1 none; relative to admission, scheduling metadata only (no preemption)
    int32_t args_off;        // word offset into the batch's args array
    int32_t args_words;      // must equal the operator's arity in kvm_ops.def
    int32_t wait_var;        // KVM_NO_VAR or a version variable: runnable only when version[wait_var] >= wait_ver
    int32_t wait_ver;
    int32_t pub_var;         // KVM_NO_VAR or a version variable: on completion version[pub_var] = pub_ver
    int32_t pub_ver;         //   (release store after a device-wide fence; pub_ver must be version[pub_var]+1 in program order)
    int32_t workload_id;     // which submitted graph this task came from (metrics only; scheduler never branches on it)
    int32_t flags;           // reserved, 0
    int32_t reserved0;
    int32_t reserved1;
};
static_assert(sizeof(KvmTaskDesc) == 64, "KVM ABI size mismatch");
static_assert(alignof(KvmTaskDesc) == 4, "KVM ABI alignment mismatch");

// Device buffer table entry. Operators address data only through table indices carried in their typed arguments.
struct KvmBuf {
    void*   p;
    int32_t rows, cols, ld;
    int32_t dtype;           // 0 = f32, 1 = i32 (index / counter buffers)
    int32_t reserved[2];
};
static_assert(sizeof(KvmBuf) == 32, "KvmBuf size mismatch");
